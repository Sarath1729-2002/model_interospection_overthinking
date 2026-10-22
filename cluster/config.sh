#!/bin/bash
# All the settings, in one place. Sourced by every script here.
# Override any of them on the command line, e.g.  SEEDS="0" bash cluster/submit.sh

# ---- fill these in ----------------------------------------------------------
# Slack: the e-mail address of the Mila Slack channel (or DM) that should get the reports.
# Slack channel -> "Integrations" -> "Send emails to this channel" gives you the address.
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
export MAX_STEPS="${MAX_STEPS:-4000}"
export TRAIN_BS="${TRAIN_BS:-8}"                          # per GPU
export EVAL_STEPS="${EVAL_STEPS:-50}"                     # repo default 10/5 re-reads ~100GB eval data 400x
export SCHED_PATIENCE="${SCHED_PATIENCE:-0}"
export WORKERS="${WORKERS:-6}"                            # dataloader workers per GPU

# SMOKE=1 -> tiny rehearsal (64 train / 32 eval prompts, 40 train steps) in its own folder.
export SMOKE="${SMOKE:-0}"
export RUN_NAME="${RUN_NAME:-${MODEL}-${DATA}}"
SMOKE_ARGS=""
if [[ "$SMOKE" == "1" ]]; then
  RUN_NAME="${RUN_NAME}-smoke"; SMOKE_ARGS="data.num_train_samples=64 data.num_eval_samples=32"
  MAX_STEPS=40; EVAL_STEPS=20; SEEDS="0"
fi
export RUN_NAME MAX_STEPS EVAL_STEPS SEEDS SMOKE_ARGS

# ---- where things live (scratch = big and fast; home has a tiny quota) -------
export ROOT="${ROOT:-${SCRATCH:-$HOME/scratch}/model_overthinking_probe}"
export VENV="${VENV:-$HOME/venvs/mop}"
export HF_HOME="$ROOT/hf_home"
export WANDB_DIR="$ROOT/wandb"
export RUN_DIR="$ROOT/outputs/$RUN_NAME"

# ---- runtime (compute nodes have no internet) -------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Slack message by e-mail. Never fails the job; always prints the subject to the log.
# Tries, in order: sendmail, mail, then Python smtplib to the node's local mail server (port 25).
# usage: notify "subject" ["body"]
notify() {
  local subject="[mop ${RUN_NAME}] $1" body="${2:-}" sm msg
  echo "[notify] $subject"
  [[ -z "$SLACK_EMAIL" ]] && return 0
  msg="$(printf 'To: %s\nFrom: %s@%s\nSubject: %s\n\n%s\n\nhost=%s job=%s\n' \
         "$SLACK_EMAIL" "${USER:-mop}" "$(hostname -f 2>/dev/null || hostname)" "$subject" "$body" "$(hostname)" "${SLURM_JOB_ID:-login}")"
  sm="$(command -v sendmail 2>/dev/null || ls /usr/sbin/sendmail /usr/lib/sendmail 2>/dev/null | head -n1 || true)"
  if [[ -n "$sm" ]] && printf '%s\n' "$msg" | "$sm" -t 2>/dev/null; then return 0; fi
  if command -v mail >/dev/null 2>&1 && printf '%s\n' "$body" | mail -s "$subject" "$SLACK_EMAIL" 2>/dev/null; then return 0; fi
  if printf '%s\n' "$msg" | python3 -c 'import smtplib,sys
from email import message_from_string
m = message_from_string(sys.stdin.read())
with smtplib.SMTP("localhost", 25, timeout=20) as s:
    s.send_message(m)' 2>/dev/null; then return 0; fi
  echo "[notify] could not send e-mail from $(hostname) (no sendmail/mail/local SMTP); message is only in this log"
  return 0
}

# torchrun on every GPU of every allocated node (1 srun task per node)
launch() {
  local gpus head port
  gpus="$(nvidia-smi -L | wc -l)"
  head="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
  port=$((20000 + SLURM_JOB_ID % 20000))
  export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / gpus > 0 ? ${SLURM_CPUS_PER_TASK:-8} / gpus : 1 ))
  srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --kill-on-bad-exit=1 \
    python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$gpus" \
      --rdzv_backend=c10d --rdzv_endpoint="$head:$port" --rdzv_id="$SLURM_JOB_ID" "$@"
}
