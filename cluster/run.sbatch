#!/bin/bash
# One job script. STAGE=extract | seed | final   (cluster/submit.sh chains them; you rarely call this by hand)
#   extract : generate answers + cache hidden states (all GPUs)
#   seed    : for ONE seed (SEED=n): train -> evaluate -> [Slack report] -> layer sweep -> [Slack report]
#   final   : combine all seeds -> final summary + layer curve with seed error bars
# Parallel: torchrun, 1 rank per GPU.
#SBATCH --job-name=mop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/config.sh"
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$RUN_DIR"
LAYER_SWEEP="${LAYER_SWEEP:-1}"
T0=$SECONDS
trap 'notify "FAILED: stage=${STAGE} ${SEED:+seed=$SEED }(line $LINENO)" "$(tail -n 25 "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null)"' ERR
elapsed() { printf '%dh%02dm' $(((SECONDS - T0) / 3600)) $((((SECONDS - T0) % 3600) / 60)); }

train_probes() {  # train_probes <run_name_suffix> <extra hydra args...>   (uses $SEED)
  local suffix="$1"; shift
  launch src/miol/train_offline.py \
    model="$MODEL" data=hidden_states data.hidden_states_dir="$RUN_DIR/hidden_states" \
    data.dataset_name="$DATA" output_dir="$RUN_DIR" \
    wandb.enabled=true wandb.run_name="${RUN_NAME}${suffix}-${SEED}" \
    seed="$SEED" max_steps="$MAX_STEPS" unified_batch_size="$TRAIN_BS" \
    eval_steps="$EVAL_STEPS" display_every_n_steps=0 \
    scheduler.scheduler_factory.patience="$SCHED_PATIENCE" \
    dataloader.num_workers="$WORKERS" dataloader.prefetch_factor=2 "$@"
}

sweep_eval() {  # sweep_eval <out_dir> <seed...>
  local out="$1"; shift; local args=() s
  for s in "$@"; do args+=(--probe-dir "$RUN_DIR/${RUN_NAME}-layers-${s}/seed_${s}/trained_probe"); done
  [[ -f "$RUN_DIR/all_layers_mae.txt" ]] && args+=(--all-layers-mae "$(cat "$RUN_DIR/all_layers_mae.txt")")
  mkdir -p "$out"
  python -m miol.analysis.layer_sweep_eval "${args[@]}" --hidden-states-dir "$RUN_DIR/hidden_states" \
    --output-dir "$out" --num-workers "$WORKERS" | tee "$out/summary.txt"
}

case "${STAGE:?set STAGE=extract|seed|final}" in
extract)
  notify "extraction STARTED" "$(nvidia-smi -L | wc -l) GPUs; max_new_tokens=$MAX_NEW_TOKENS; smoke=$SMOKE"
  launch src/miol/extract_hidden_states.py \
    model="$MODEL" data="$DATA" output_dir="$RUN_DIR" seed=0 \
    generation.max_new_tokens="$MAX_NEW_TOKENS" \
    extract.batch_size="$GEN_BATCH" extract.forward_batch_size="$FWD_BATCH" \
    wandb.enabled=false $SMOKE_ARGS
  python cluster/audit_extraction.py "$RUN_DIR/hidden_states" | tee "$RUN_DIR/hidden_states/audit.txt"
  notify "extraction DONE in $(elapsed)" "$(cat "$RUN_DIR/hidden_states/audit.txt")
disk: $(du -sh "$RUN_DIR/hidden_states" | cut -f1)"
  ;;
seed)
  SEED="${SEED:?set SEED=<n>}"
  d="$RUN_DIR/${RUN_NAME}-${SEED}/seed_${SEED}"
  # 1) main probes (all layers concatenated), then score them -> first Slack report
  notify "seed $SEED: training STARTED"
  train_probes "" data.hidden_layer=all wandb.project=mop-"$MODEL"
  python src/miol/evaluate_probes.py --probe-dir "$d/trained_probe" \
    --hidden-states-dir "$RUN_DIR/hidden_states" --output "$d/eval_results.json"
  python cluster/summarize_eval.py "$RUN_DIR" "$SEED" | tee "$RUN_DIR/summary_seed${SEED}.txt"
  notify "seed $SEED: main results ready (elapsed $(elapsed))" "$(cat "$RUN_DIR/summary_seed${SEED}.txt")"
  # 2) per-layer sweep (one head per layer), then score it -> second Slack report
  if [[ "$LAYER_SWEEP" == "1" ]]; then
    notify "seed $SEED: layer sweep STARTED"
    train_probes "-layers" data.layers_to_train=all wandb.project=mop-"$MODEL"-layers \
      'optimization.probes=[count,prompt_only_count,baseline_count,baseline_position_count]'
    sweep_eval "$RUN_DIR/layer_sweep_seed${SEED}" "$SEED"
    notify "seed $SEED: layer curve ready (elapsed $(elapsed))" "$(cat "$RUN_DIR/layer_sweep_seed${SEED}/summary.txt")"
  fi
  ;;
final)
  python cluster/summarize_eval.py "$RUN_DIR" | tee "$RUN_DIR/summary.txt"
  if [[ "$LAYER_SWEEP" == "1" ]]; then
    sweep_eval "$RUN_DIR/layer_sweep" $SEEDS
    body="$(cat "$RUN_DIR/summary.txt")

$(cat "$RUN_DIR/layer_sweep/summary.txt")"
  else
    body="$(cat "$RUN_DIR/summary.txt")"
  fi
  notify "ALL SEEDS DONE - final results (seeds: $SEEDS)" "$body"
  ;;
*) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
