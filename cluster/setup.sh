#!/bin/bash
# ONE-TIME setup. Run on the LOGIN node (it has internet) from the repo root:
#   bash cluster/setup.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0   # download needs the network

module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || echo "[warn] module load failed (StdEnv/2023 gcc arrow python/3.12) - run: module spider arrow python"
mkdir -p "$ROOT/outputs" "$HF_HOME" "$WANDB_DIR" "$(dirname "$VENV")"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install --no-index "torch>=2.9.0" 2>/dev/null || pip install "torch>=2.9.0"   # cluster wheel first
# pyarrow comes from the cluster's "arrow" module (loaded above); the plain wheel is a stub that fails to build
pip install -e .
ln -sfn "$ROOT/outputs" outputs

echo "== downloading $MODEL_ID and $DATA into $HF_HOME =="
python - <<'PY'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download(os.environ["MODEL_ID"], allow_patterns=["*.json", "*.safetensors", "*.txt"])
if os.environ["DATA"] == "gsm8k":
    load_dataset("openai/gsm8k", "main", split="train"); load_dataset("openai/gsm8k", "main", split="test")
print("download ok")
PY

echo "== offline check (what the compute nodes will see) =="
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import os
from transformers import AutoTokenizer, GenerationConfig
m = os.environ["MODEL_ID"]; t = AutoTokenizer.from_pretrained(m)
print("eos:", t.eos_token, t.eos_token_id, "| pad:", t.pad_token, t.pad_token_id)
print("checkpoint generation_config:", GenerationConfig.from_pretrained(m))
PY

[[ -z "$SLACK_EMAIL" ]] && echo "[todo] SLACK_EMAIL is empty - set it in cluster/config.sh to get Slack reports"
[[ -z "$SBATCH_ACCOUNT" ]] && echo "[todo] SBATCH_ACCOUNT is empty - set it in cluster/config.sh (e.g. def-yourpi)"
notify "setup finished on the login node" "venv=$VENV  cache=$HF_HOME  outputs=$ROOT/outputs"
echo "Setup done. Next:  SMOKE=1 bash cluster/submit.sh"
