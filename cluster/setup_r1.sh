#!/bin/bash
# Run ONCE on the LOGIN node (needs internet) for the DeepSeek-R1-Distill-Qwen-7B run.
#   bash cluster/setup_r1.sh
# 1. downloads the model into the HF cache   2. copies the problem lists of the Qwen MATH run, so both models see the SAME problems
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
unset RUN_NAME RUN_DIR
export DATA=math MODEL=r1_distill_7b MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" MSIG_STYLE=r1
source cluster/config.sh
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
QWEN_RUN="$ROOT/outputs/qwen2.5_7b-math"
[[ -f "$QWEN_RUN/problems_test.jsonl" ]] || { echo "missing $QWEN_RUN/problems_test.jsonl - run the Qwen MATH setup first"; exit 1; }
mkdir -p "$RUN_DIR"
cp -n "$QWEN_RUN"/problems_train.jsonl "$QWEN_RUN"/problems_test.jsonl "$RUN_DIR"/
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
print("model cached at", p)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("$MODEL_ID")
s = t.apply_chat_template([{"role": "user", "content": "What is 1+1?"}], tokenize=False, add_generation_prompt=True)
print("chat template renders:", repr(s))
print("eos:", t.eos_token, t.eos_token_id)
PY
wc -l "$RUN_DIR"/problems_*.jsonl
echo "run dir: $RUN_DIR"
echo "next:  bash cluster/submit_r1.sh   (MODE=gate first)"
