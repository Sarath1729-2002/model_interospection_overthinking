#!/bin/bash
# Submit the MATH signals experiment. Run from the repo root on the login node AFTER  bash cluster/setup_math.sh
#   MODE=gate bash cluster/submit_math.sh     # ~20-30 min: is the signal there on MATH?  READ THE GATE FIRST
#   MODE=full bash cluster/submit_math.sh     # work -> merge -> [rollouts -> relabel] -> probes -> controls -> analyze -> transfer -> steer
#   MODE=extras bash cluster/submit_math.sh   # only transfer + steer (after a finished full run)
# Resubmitting is safe: finished chunks are skipped.  MSIG_ROLLOUTS=1 adds the resampled-continuation arm.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
unset RUN_NAME RUN_DIR
export DATA=math
source cluster/config.sh
[[ -z "$SBATCH_ACCOUNT" ]] && { echo "Set SBATCH_ACCOUNT in cluster/config.sh"; exit 1; }
[[ -f "$RUN_DIR/problems_test.jsonl" ]] || { echo "run bash cluster/setup_math.sh on the login node first"; exit 1; }
MODE="${MODE:-gate}"
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --gpus-per-node="$GPUS_PER_NODE" --mem="$MEM")
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
LIGHT=(--gpus-per-node="$FINAL_GPUS" --cpus-per-task=16)
sub() { # sub <stage> <time> <dependency-or-empty> [extra sbatch args...]
  local stage="$1" t="$2" dep="$3"; shift 3
  local dargs=(); [[ -n "$dep" ]] && dargs=(--dependency=afterok:"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="msig-$stage" --time="$t" ${dargs[@]+"${dargs[@]}"} "$@" cluster/run_math.sbatch
}
if [[ "$MODE" == "gate" ]]; then
  j=$(sub gate 01:00:00 "")
  echo "submitted gate=$j   (result: Slack + $RUN_DIR/signals_gate/gate_report_eval.txt)"; notify "math gate SUBMITTED" "gate=$j"; exit 0
fi
if [[ "$MODE" == "extras" ]]; then
  t=$(sub transfer 00:30:00 "" "${LIGHT[@]}"); s=$(sub steer 00:40:00 "" "${LIGHT[@]}")
  echo "submitted transfer=$t steer=$s"; exit 0
fi
wk=$(sub work 24:00:00 "")
mg=$(sub merge 00:45:00 "$wk" "${LIGHT[@]}")
dep="$mg"; msg="work=$wk merge=$mg"
if [[ "$MSIG_ROLLOUTS" == "1" ]]; then
  ro=$(sub rollouts 05:00:00 "$mg"); rl=$(sub relabel 00:30:00 "$ro" "${LIGHT[@]}"); dep="$rl"; msg="$msg rollouts=$ro relabel=$rl"
fi
pr=$(sub probes 04:00:00 "$dep" "${LIGHT[@]}")
co=$(sub controls 01:30:00 "$pr" "${LIGHT[@]}")
an=$(sub analyze 00:45:00 "$pr" "${LIGHT[@]}")
tr=""; [[ "$MSIG_TRANSFER" == "1" ]] && tr=$(sub transfer 00:30:00 "$pr" "${LIGHT[@]}")
st=$(sub steer 00:40:00 "$pr" "${LIGHT[@]}")
msg="$msg probes=$pr controls=$co analyze=$an transfer=${tr:-skipped} steer=$st"
echo "submitted: $msg   (watch: squeue -u \$USER)"; notify "math pipeline SUBMITTED" "$msg"
