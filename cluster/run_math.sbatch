#!/bin/bash
# Job script for the MATH signals experiment. STAGE = gate | work | merge | rollouts | relabel | probes | controls | analyze | transfer | steer
#   gate     : generate + score + states for MSIG_GATE_N test problems (own folder signals_gate/), merge, gate report
#   work     : generate -> checkpoint states + token stats -> forced-answer scores, test then train   1 process per GPU, resumable
#   merge    : concatenate chunks, derive labels, gate reports                                       CPU
#   rollouts : resampled continuations (optional)                                                    1 process per GPU, resumable
#   relabel  : re-derive labels after the rollouts (states untouched)                                CPU
#   probes   : every ridge probe + baselines (B0..B5, MLPs), all layers + all-layer kernel            1 GPU
#   controls : Q0 (question-only state) + stacking                                                   1 GPU
#   analyze  : tables, plots, report.md                                                              CPU
#   transfer : MATH-trained probes applied to the GSM8K states                                       1 GPU
#   steer    : add the probe direction to the residual stream, measure the change in log P(gold)     1 GPU
#SBATCH --job-name=msig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
export DATA=math
source "${SLURM_SUBMIT_DIR}/cluster/config.sh"
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
SIG="$RUN_DIR/signals"
mkdir -p "$SIG"
T0=$SECONDS
trap 'notify "FAILED: math stage=${STAGE} (line $LINENO)" "$(tail -n 25 "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null)"' ERR
elapsed() { printf '%dh%02dm' $(((SECONDS - T0) / 3600)) $((((SECONDS - T0) % 3600) / 60)); }

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
WORK=(--run-dir "$RUN_DIR" --out "$SIG" --model-id "$MODEL_ID" --max-new "$MSIG_MAX_NEW" --chunk-problems "$MSIG_CHUNK")
PROBE_ARGS=(--run-dir "$RUN_DIR" --out "$SIG" --val-fraction "$VAL_FRACTION" --split-source labels)
[[ -f "$RUN_DIR/problems_test.jsonl" ]] || { echo "no problem lists: run bash cluster/setup_math.sh on the login node first"; exit 1; }

case "${STAGE:?set STAGE}" in
gate)
  G="$RUN_DIR/signals_gate"; mkdir -p "$G"
  notify "math GATE started" "$MSIG_GATE_N test problems"
  run_sharded cluster/msig_worker.py --run-dir "$RUN_DIR" --out "$G" --model-id "$MODEL_ID" --max-new "$MSIG_MAX_NEW" --chunk-problems 40 --split test --limit "$MSIG_GATE_N"
  python cluster/msig_merge.py --run-dir "$RUN_DIR" --out "$G" --splits test
  notify "math GATE done in $(elapsed)" "$(cat "$G/gate_report_eval.txt")"
  ;;
work)
  notify "math: worker STARTED" "$(nvidia-smi -L | wc -l) GPUs; test=$MSIG_TEST_N train=$MSIG_TRAIN_N"
  run_sharded cluster/msig_worker.py "${WORK[@]}" --split test --limit "$MSIG_TEST_N"
  run_sharded cluster/msig_worker.py "${WORK[@]}" --split train --limit "$MSIG_TRAIN_N"
  notify "math: worker DONE in $(elapsed)"
  ;;
merge)
  python cluster/msig_merge.py --run-dir "$RUN_DIR" --out "$SIG" --tau "${SIG_TAU:-30}"
  notify "math: merged + labelled" "$(cat "$SIG/gate_report_eval.txt")"
  ;;
rollouts)
  notify "math: rollouts STARTED"
  RA=(--run-dir "$RUN_DIR" --out "$SIG" --model-id "$MODEL_ID" --data math --m "$MSIG_ROLL_M" --max-new "$MSIG_ROLL_MAX_NEW" --rows-per-batch 24)
  run_sharded cluster/sig_rollouts.py "${RA[@]}" --split eval --n-traces "$MSIG_ROLL_EVAL"
  run_sharded cluster/sig_rollouts.py "${RA[@]}" --split train --n-traces "$MSIG_ROLL_TRAIN"
  notify "math: rollouts DONE in $(elapsed)"
  ;;
relabel)
  python cluster/msig_merge.py --run-dir "$RUN_DIR" --out "$SIG" --tau "${SIG_TAU:-30}" --labels-only
  ;;
probes)
  notify "math: probes STARTED"
  python cluster/sig_probes.py "${PROBE_ARGS[@]}" --kernel-max-fit "$SIG_KERNEL_MAX_FIT" --save-weights
  notify "math: probes DONE in $(elapsed)" "$(python - "$SIG/probes/scores_test.csv" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
t = d[d.target.isin(["a", "f", "a_next", "d_gold", "rem"])].pivot(index="model", columns="target", values="r2")
keep = [m for m in ["B0", "B1", "B3_mlp", "B4", "B5", "B5_mlp", "ALL"] if m in t.index] + [m for m in t.index if m.startswith("L")][::7]
print("test R2\n" + t.loc[keep].round(3).to_string())
PY
)"
  ;;
controls)
  python cluster/sig_controls.py --run-dir "$RUN_DIR" --out "$SIG" --val-fraction "$VAL_FRACTION" --split-source labels --no-bq --layers "${SIG_CONTROL_LAYERS:-16,20,22}"
  notify "math: controls DONE in $(elapsed)" "$(grep -E 'ADDED VALUE' "$SIG/analysis/controls.md" | cut -c1-200 | head -24)"
  ;;
analyze)
  python cluster/sig_analyze.py --run-dir "$RUN_DIR" --out "$SIG"
  notify "math: ANALYSIS ready" "$(sed -n 1,40p "$SIG/analysis/report.md")"
  ;;
transfer)
  python cluster/msig_transfer.py --math-out "$SIG" --gsm-out "$GSM_RUN_DIR/signals" --layers "${SIG_TRANSFER_LAYERS:-12,16,20,22,24,26}"
  notify "math: transfer DONE" "$(sed -n 1,30p "$SIG/analysis/transfer.md")"
  ;;
steer)
  python cluster/msig_steer.py --run-dir "$RUN_DIR" --out "$SIG" --model-id "$MODEL_ID" --layer "$MSIG_STEER_LAYER"
  notify "math: steering DONE" "$(sed -n 1,30p "$SIG/analysis/steer.md")"
  ;;
*) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
