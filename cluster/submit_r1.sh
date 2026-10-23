#!/bin/bash
# DeepSeek-R1-Distill-Qwen-7B run on the same MATH problems as the Qwen2.5-7B-Instruct run.
#   MODE=gate bash cluster/submit_r1.sh    # ~45 min: truncation rate, accuracy, does the forced-answer cue work on R1?  READ THIS FIRST
#   MODE=full bash cluster/submit_r1.sh
# then compare with:  python cluster/msig_compare.py   (see GUIDE_R1.md)
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
export MODEL=r1_distill_7b MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" MSIG_STYLE=r1 DATA=math
exec bash cluster/submit_math.sh
