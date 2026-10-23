#!/bin/bash
# All the settings, in one place. Sourced by every script here.
# Override any of them on the command line, e.g.  SEEDS="0" bash cluster/submit.sh

# ---- fill these in ----------------------------------------------------------
# Slack/phone reports: set SLACK_WEBHOOK_URL and/or NTFY_TOPIC in ~/.bashrc (see notify() below).
# SLACK_EMAIL (your Slackbot forwarding address) receives Slurm's own BEGIN/END/FAIL mails for every job.
export SLACK_EMAIL="${SLACK_EMAIL:-}"
# Your allocation, e.g. def-oberman (see: sacctmgr show assoc user=$USER format=account%30)
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

# ---- what to run ------------------------------------------------------------
export MODEL="${MODEL:-qwen2.5_7b}"                       # src/miol/conf/model/
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export DATA="${DATA:-gsm8k}"                              # src/miol/conf/data/
export SEEDS="${SEEDS:-0 1 2}"                            # probe seeds (paper: mean of 3)
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"           # repo default 300 would truncate GSM8K answers
export GEN_BATCH="${GEN_BATCH:-32}"                       # prompts per generate() call, per GPU
export FWD_BATCH="${FWD_BATCH:-16}"                       # sequences per hidden-state forward, per GPU
export MAX_STEPS="${MAX_STEPS:-4000}"                     # hard cap; early stopping usually ends the run sooner
export TRAIN_BS="${TRAIN_BS:-8}"                          # per GPU
export EVAL_STEPS="${EVAL_STEPS:-100}"                    # each eval re-reads the validation samples from disk
export SCHED_PATIENCE="${SCHED_PATIENCE:-1}"              # evals without improvement before the LR halves
# Validation split + early stopping: hold VAL_FRACTION of TRAIN out (same split for every seed); the LR
# schedule, early stopping and best-step selection use ONLY that split, the test split is scored once at the end.
export VAL_FRACTION="${VAL_FRACTION:-0.05}"
export ES_ENABLED="${ES_ENABLED:-1}"                      # 0 = train for exactly MAX_STEPS
export ES_PATIENCE="${ES_PATIENCE:-5}"                    # evals without >ES_MIN_DELTA improvement before stopping
export ES_MIN_DELTA="${ES_MIN_DELTA:-0.25}"               # tokens of MAE
# Ridge cross-check (does a closed-form fit beat the trained probe? -> is the probe optimisation-limited?)
export RIDGE="${RIDGE:-1}"
export RIDGE_LAYERS="${RIDGE_LAYERS:-8,12,16,20,24,26,27,28}"   # or "all"
# Slurm resources (TamIA hands out H100s by whole node: 4 per node). On Fir use e.g. GPUS_PER_NODE=h100:4 MEM=240G
export GPUS_PER_NODE="${GPUS_PER_NODE:-h100:4}"
export MEM="${MEM:-0}"
export FINAL_GPUS="${FINAL_GPUS:-$GPUS_PER_NODE}"         # for the light jobs (final summary, ridge); use h100:1 where partial nodes are allowed
export WORKERS="${WORKERS:-6}"                            # dataloader workers per GPU

# SMOKE=1 -> tiny rehearsal (64 train / 32 eval prompts, 40 train steps) in its own folder.
export SMOKE="${SMOKE:-0}"
export RUN_NAME="${RUN_NAME:-${MODEL}-${DATA}}"
SMOKE_ARGS=""
if [[ "$SMOKE" == "1" ]]; then
  [[ "$RUN_NAME" == *-smoke ]] || RUN_NAME="${RUN_NAME}-smoke"; SMOKE_ARGS="data.num_train_samples=64 data.num_eval_samples=32"
  MAX_STEPS=40; EVAL_STEPS=20; SEEDS="0"
fi
export RUN_NAME MAX_STEPS EVAL_STEPS SEEDS SMOKE_ARGS

# ---- where things live (scratch = big and fast; home has a tiny quota) -------
export ROOT="${ROOT:-${SCRATCH:-$HOME/scratch}/model_overthinking_probe}"
export VENV="${VENV:-$HOME/venvs/mop}"
export HF_HOME="$ROOT/hf_home"
export WANDB_DIR="$ROOT/wandb"
export RUN_DIR="$ROOT/outputs/$RUN_NAME"

# ---- signals experiment (cluster/submit_signals.sh) ---------------------------------------------------------------
# Forced-answer probes on the cached GSM8K run: needs RUN_DIR/hidden_states from the extract stage. Outputs: $RUN_DIR/signals/
export SIG_MODE="${SIG_MODE:-lines}"                      # checkpoints: lines = every blank-line boundary; both = + every SIG_STRIDE tokens
export SIG_STRIDE="${SIG_STRIDE:-8}"
export SIG_TAU="${SIG_TAU:-30}"                           # tokens written after convergence/commitment that make it a 'condition'
export SIG_LIMIT_TRAIN="${SIG_LIMIT_TRAIN:-0}"            # 0 = all cached train traces
export SIG_LIMIT_EVAL="${SIG_LIMIT_EVAL:-0}"
export SIG_GATE_N="${SIG_GATE_N:-300}"                    # eval traces scored by the feasibility gate
export SIG_ROLLOUTS="${SIG_ROLLOUTS:-1}"                  # 1 = also resample continuations (validation + extra targets)
export SIG_ROLL_TRAIN="${SIG_ROLL_TRAIN:-2000}"           # traces that get rollouts (train / eval)
export SIG_ROLL_EVAL="${SIG_ROLL_EVAL:-600}"
export SIG_ROLL_M="${SIG_ROLL_M:-4}"                      # continuations per checkpoint
export SIG_KERNEL_MAX_FIT="${SIG_KERNEL_MAX_FIT:-50000}"  # rows used by the all-layer dual ridge (memory: ~3 n^2 floats)
export SIG_STATES_WORKERS="${SIG_STATES_WORKERS:-8}"

# ---- runtime (compute nodes have no internet) -------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Stage reports. Never fails the job; always prints the subject to the log.
# Channels used if configured (put the secrets in ~/.bashrc as `export ...`, never in the repo):
#   SLACK_WEBHOOK_URL  Slack incoming webhook posting to your DM   (works from Fir/Nibi compute nodes: they have internet)
#   NTFY_TOPIC         private topic name for push notifications via ntfy.sh (phone app)
# If neither is set it falls back to e-mail to SLACK_EMAIL (sendmail, mail, local SMTP) - not available on Fir nodes.
# usage: notify "subject" ["body"]
notify() {
  local subject="[mop ${RUN_NAME}] $1" body="${2:-}" text sent=0 sm msg
  echo "[notify] $subject"
  text="*${subject}*"$'\n'"${body}"$'\n'"host=$(hostname) job=${SLURM_JOB_ID:-login}"
  if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    if printf '%s' "${text:0:30000}" | python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' \
         | curl -sS --fail -m 20 -X POST -H 'Content-type: application/json' --data @- "$SLACK_WEBHOOK_URL" >/dev/null 2>&1; then
      sent=1
    else echo "[notify] Slack webhook post failed"; fi
  fi
  if [[ -n "${NTFY_TOPIC:-}" ]]; then
    if curl -sS --fail -m 20 -H "Title: ${subject}" -d "${text:0:3500}" "${NTFY_SERVER:-https://ntfy.sh}/${NTFY_TOPIC}" >/dev/null 2>&1; then
      sent=1
    else echo "[notify] ntfy post failed"; fi
  fi
  [[ $sent == 1 || -z "${SLACK_EMAIL:-}" ]] && return 0
  msg="$(printf 'To: %s\nFrom: %s@%s\nSubject: %s\n\n%s\n' "$SLACK_EMAIL" "${USER:-mop}" "$(hostname)" "$subject" "$text")"
  sm="$(command -v sendmail 2>/dev/null || ls /usr/sbin/sendmail /usr/lib/sendmail 2>/dev/null | head -n1 || true)"
  if [[ -n "$sm" ]] && printf '%s\n' "$msg" | "$sm" -t 2>/dev/null; then return 0; fi
  if command -v mail >/dev/null 2>&1 && printf '%s\n' "$text" | mail -s "$subject" "$SLACK_EMAIL" 2>/dev/null; then return 0; fi
  if printf '%s\n' "$msg" | python3 -c 'import smtplib,sys
from email import message_from_string
m = message_from_string(sys.stdin.read())
with smtplib.SMTP("localhost", 25, timeout=20) as s:
    s.send_message(m)' 2>/dev/null; then return 0; fi
  echo "[notify] could not deliver the message anywhere; it is only in this log"
  return 0
}

# torchrun on every GPU of every allocated node (1 srun task per node)
launch() {
  local gpus head port
  gpus="$(nvidia-smi -L | wc -l)"
  head="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
  port=$((20000 + SLURM_JOB_ID % 20000))
  if [[ "${SLURM_NNODES:-1}" == "1" ]]; then RDZV_ARGS="--standalone"; else RDZV_ARGS="--rdzv_backend=c10d --rdzv_endpoint=$head:$port --rdzv_id=$SLURM_JOB_ID"; fi
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / gpus > 0 ? ${SLURM_CPUS_PER_TASK:-8} / gpus : 1 ))
  srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --kill-on-bad-exit=1 \
    python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$gpus" \
      $RDZV_ARGS "$@"
}

# ---- MATH signals experiment (cluster/setup_math.sh, cluster/submit_math.sh) ----------------------------------------
# Own run folder (DATA=math -> $ROOT/outputs/${MODEL}-math): generation, forced-answer scoring and hidden states are all done in
# ONE pass by cluster/msig_worker.py, so no cached extraction is needed.  GSM8K run (for the transfer test): $GSM_RUN_DIR/signals.
# MSIG_STYLE=qwen (default; Qwen2.5-Instruct: system prompt, short solutions) | r1 (DeepSeek-R1-Distill: no system prompt, <think> block,
# sampling T=0.6/top_p=0.95, long solutions -> bigger cap, smaller generation batches, capped checkpoints per trace)
export MSIG_STYLE="${MSIG_STYLE:-qwen}"
if [[ "$MSIG_STYLE" == "r1" ]]; then
  MSIG_MAX_NEW="${MSIG_MAX_NEW:-6144}"; MSIG_TEMP="${MSIG_TEMP:-0.6}"; MSIG_TOP_P="${MSIG_TOP_P:-0.95}"; MSIG_TOP_K="${MSIG_TOP_K:-0}"
  MSIG_GEN_BATCH="${MSIG_GEN_BATCH:-32}"; MSIG_MAX_CKPT="${MSIG_MAX_CKPT:-40}"; MSIG_TRAIN_N="${MSIG_TRAIN_N:-2000}"; MSIG_CHUNK="${MSIG_CHUNK:-64}"
  MSIG_TRANSFER="${MSIG_TRANSFER:-0}"
fi
export MSIG_TEMP="${MSIG_TEMP:-0.7}" MSIG_TOP_P="${MSIG_TOP_P:-0.8}" MSIG_TOP_K="${MSIG_TOP_K:-20}" MSIG_GEN_BATCH="${MSIG_GEN_BATCH:-0}"
export MSIG_MAX_CKPT="${MSIG_MAX_CKPT:-0}" MSIG_TRANSFER="${MSIG_TRANSFER:-1}"
export MSIG_CUE="${MSIG_CUE:-}"                           # empty = the default cue (same for both models, so the runs are comparable)
export MSIG_TRAIN_N="${MSIG_TRAIN_N:-0}"                  # train problems used (0 = every integer-answer MATH train problem)
export MSIG_TEST_N="${MSIG_TEST_N:-1000}"                 # held-out problems (0 = all)
export MSIG_GATE_N="${MSIG_GATE_N:-160}"                  # problems of the quick gate
export MSIG_MAX_NEW="${MSIG_MAX_NEW:-2048}"               # generation cap (traces that hit it are flagged as truncated)
export MSIG_CHUNK="${MSIG_CHUNK:-96}"                     # problems per resumable chunk
export MSIG_ROLLOUTS="${MSIG_ROLLOUTS:-0}"                # 1 = resampled continuations (slow: ~1.5-2.5 h); weak on GSM8K, off by default
export MSIG_ROLL_EVAL="${MSIG_ROLL_EVAL:-150}"
export MSIG_ROLL_TRAIN="${MSIG_ROLL_TRAIN:-300}"
export MSIG_ROLL_M="${MSIG_ROLL_M:-3}"
export MSIG_ROLL_MAX_NEW="${MSIG_ROLL_MAX_NEW:-1200}"
export MSIG_STEER_LAYER="${MSIG_STEER_LAYER:-20}"
export GSM_RUN_DIR="${GSM_RUN_DIR:-$ROOT/outputs/${MODEL}-gsm8k}"
