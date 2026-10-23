#!/bin/bash
# Run ONCE on the LOGIN node (needs internet): downloads MATH + the model into the HF cache and writes the problem lists.
#   bash cluster/setup_math.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
export DATA=math
source cluster/config.sh
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
module purge 2>/dev/null || true
module load StdEnv/2023 gcc arrow python/3.12 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR"
python cluster/msig_data.py --run-dir "$RUN_DIR"
python - <<PY
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("$MODEL_ID")      # model weights are already cached by the GSM8K run; this only checks the tokenizer
print("tokenizer ok")
PY
wc -l "$RUN_DIR"/problems_*.jsonl
echo "next:  MODE=gate bash cluster/submit_math.sh"
