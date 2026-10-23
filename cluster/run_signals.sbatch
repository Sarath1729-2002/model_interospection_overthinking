#!/bin/bash
# One job script for the signals experiment. STAGE = gate | ckpt | score | states | rollouts | derive | probes | analyze
# (cluster/submit_signals.sh chains them.)  Everything reads the cached run in $RUN_DIR/hidden_states and writes $RUN_DIR/signals/.
#   gate     : feasibility check on SIG_GATE_N eval traces (checkpoints -> score -> derive -> gate report), own folder signals_gate/
#   ckpt     : checkpoint tables (+ exact labels that need no model)              CPU
#   score    : forced-answer log-probs + greedy answer at every checkpoint        1 process per GPU, resumable
#   states   : gather the hidden state of every checkpoint into one fp16 array    I/O bound
#   rollouts : resampled continuations -> roll_H / roll_pgold / roll_pfinal       1 process per GPU, resumable
#   derive   : merge scores, derive convergence/commitment labels + conditions, print the gate report
#   probes   : all ridge probes + baselines                                       1 GPU
#   analyze  : tables, plots, report.md                                           CPU
#SBATCH --job-name=sig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/config.sh"
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
SIG="$RUN_DIR/signals"
mkdir -p "$SIG"
T0=$SECONDS
trap 'notify "FAILED: signals stage=${STAGE} (line $LINENO)" "$(tail -n 25 "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null)"' ERR
elapsed() { printf '%dh%02dm' $(((SECONDS - T0) / 3600)) $((((SECONDS - T0) % 3600) / 60)); }

# run_sharded <script> <args...> : one process per GPU, --shard i --nshards n appended; fails if any shard fails
run_sharded() {
  local n i rc=0 pids=()
  n="$(nvidia-smi -L | wc -l)"
  for ((i = 0; i < n; i++)); do
    CUDA_VISIBLE_DEVICES=$i python "$@" --shard "$i" --nshards "$n" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
LIM_ARGS_TRAIN=(); LIM_ARGS_EVAL=()
[[ "$SIG_LIMIT_TRAIN" != "0" ]] && LIM_ARGS_TRAIN=(--limit "$SIG_LIMIT_TRAIN")
[[ "$SIG_LIMIT_EVAL" != "0" ]] && LIM_ARGS_EVAL=(--limit "$SIG_LIMIT_EVAL")

case "${STAGE:?set STAGE=gate|ckpt|score|states|rollouts|derive|probes|analyze}" in
gate)
  G="$RUN_DIR/signals_gate"; mkdir -p "$G"
  notify "signals GATE started" "$SIG_GATE_N eval traces"
  python cluster/sig_checkpoints.py --run-dir "$RUN_DIR" --out "$G" --splits eval --mode "$SIG_MODE" --stride "$SIG_STRIDE" --limit "$SIG_GATE_N"
  run_sharded cluster/sig_score.py --run-dir "$RUN_DIR" --out "$G" --split eval --model-id "$MODEL_ID"
  python cluster/sig_derive.py --run-dir "$RUN_DIR" --out "$G" --splits eval --tau "$SIG_TAU"
  notify "signals GATE done in $(elapsed)" "$(cat "$G/gate_report_eval.txt")"
  ;;
ckpt)
  notify "signals: checkpoints STARTED"
  python cluster/sig_checkpoints.py --run-dir "$RUN_DIR" --out "$SIG" --splits train --mode "$SIG_MODE" --stride "$SIG_STRIDE" ${LIM_ARGS_TRAIN[@]+"${LIM_ARGS_TRAIN[@]}"}
  python cluster/sig_checkpoints.py --run-dir "$RUN_DIR" --out "$SIG" --splits eval --mode "$SIG_MODE" --stride "$SIG_STRIDE" ${LIM_ARGS_EVAL[@]+"${LIM_ARGS_EVAL[@]}"}
  notify "signals: checkpoints DONE in $(elapsed)" "train $(($(wc -l < "$SIG/ckpt_train.csv") - 1)) rows, eval $(($(wc -l < "$SIG/ckpt_eval.csv") - 1)) rows"
  ;;
score)
  notify "signals: scoring STARTED" "$(nvidia-smi -L | wc -l) GPUs"
  run_sharded cluster/sig_score.py --run-dir "$RUN_DIR" --out "$SIG" --split eval --model-id "$MODEL_ID"
  run_sharded cluster/sig_score.py --run-dir "$RUN_DIR" --out "$SIG" --split train --model-id "$MODEL_ID"
  notify "signals: scoring DONE in $(elapsed)"
  ;;
states)
  python cluster/sig_states.py --run-dir "$RUN_DIR" --out "$SIG" --split eval --workers "$SIG_STATES_WORKERS"
  python cluster/sig_states.py --run-dir "$RUN_DIR" --out "$SIG" --split train --workers "$SIG_STATES_WORKERS"
  notify "signals: states gathered in $(elapsed)" "$(du -sh "$SIG"/states_*.npy | tr '\n' ' ')"
  ;;
rollouts)
  notify "signals: rollouts STARTED" "M=$SIG_ROLL_M  eval traces=$SIG_ROLL_EVAL  train traces=$SIG_ROLL_TRAIN"
  run_sharded cluster/sig_rollouts.py --run-dir "$RUN_DIR" --out "$SIG" --split eval --model-id "$MODEL_ID" --n-traces "$SIG_ROLL_EVAL" --m "$SIG_ROLL_M"
  run_sharded cluster/sig_rollouts.py --run-dir "$RUN_DIR" --out "$SIG" --split train --model-id "$MODEL_ID" --n-traces "$SIG_ROLL_TRAIN" --m "$SIG_ROLL_M"
  notify "signals: rollouts DONE in $(elapsed)"
  ;;
derive)
  python cluster/sig_derive.py --run-dir "$RUN_DIR" --out "$SIG" --tau "$SIG_TAU"
  notify "signals: labels derived (gate report)" "$(cat "$SIG/gate_report_eval.txt")"
  ;;
probes)
  notify "signals: probes STARTED"
  python cluster/sig_probes.py --run-dir "$RUN_DIR" --out "$SIG" --val-fraction "$VAL_FRACTION" --kernel-max-fit "$SIG_KERNEL_MAX_FIT"
  notify "signals: probes DONE in $(elapsed)" "$(python - "$SIG/probes/scores_test.csv" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
t = d[d.target.isin(["a", "f", "d_gold", "rem"])].pivot(index="model", columns="target", values="r2")
keep = [m for m in ["B0", "B1", "B3_mlp", "B4", "ALL"] if m in t.index] + [m for m in t.index if m.startswith("L")][::7]
print("test R2\n" + t.loc[keep].round(3).to_string())
PY
)"
  ;;
analyze)
  python cluster/sig_analyze.py --run-dir "$RUN_DIR" --out "$SIG"
  notify "signals: ANALYSIS ready" "$(sed -n 1,30p "$SIG/analysis/report.md")"
  ;;
*) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
