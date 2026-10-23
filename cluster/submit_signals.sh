#!/bin/bash
# Submit the signals experiment on top of an existing extraction.  Run from the repo root on the login node.
#   MODE=gate bash cluster/submit_signals.sh     # ~15 min: is the forced-answer signal there at all?  READ THE GATE FIRST
#   MODE=full bash cluster/submit_signals.sh     # the whole thing: ckpt -> score || states -> [rollouts] -> derive -> probes -> analyze
#   SMOKE=1 MODE=full bash cluster/submit_signals.sh   # rehearsal on the -smoke run (run `SMOKE=1 bash cluster/submit.sh` first)
#   ROLLOUTS: SIG_ROLLOUTS=0 skips the resampled-continuation arm (saves ~1-1.5 h).
# Needs $RUN_DIR/hidden_states (the extract stage of cluster/submit.sh). Resubmitting is safe: finished chunks are skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
[[ -z "$SBATCH_ACCOUNT" ]] && { echo "Set SBATCH_ACCOUNT in cluster/config.sh"; exit 1; }
[[ -d "$RUN_DIR/hidden_states/train" ]] || { echo "no cached run at $RUN_DIR/hidden_states - run cluster/submit.sh first"; exit 1; }
MODE="${MODE:-gate}"
COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL --gpus-per-node="$GPUS_PER_NODE" --mem="$MEM")
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=FAIL,TIME_LIMIT_80)
LIGHT=(--gpus-per-node="$FINAL_GPUS" --cpus-per-task=16)
if [[ "$SMOKE" == "1" ]]; then T_GATE=00:30:00 T_CKPT=00:20:00 T_SCORE=00:40:00 T_STATES=00:30:00 T_ROLL=00:40:00 T_DERIVE=00:15:00 T_PROBE=00:40:00 T_AN=00:20:00
else T_GATE=01:00:00 T_CKPT=01:30:00 T_SCORE=04:00:00 T_STATES=05:00:00 T_ROLL=05:00:00 T_DERIVE=00:30:00 T_PROBE=03:00:00 T_AN=00:45:00; fi

sub() { # sub <stage> <time> <dependency-or-empty> [extra sbatch args...]
  local stage="$1" t="$2" dep="$3"; shift 3
  local dargs=(); [[ -n "$dep" ]] && dargs=(--dependency=afterok:"$dep")
  STAGE="$stage" sbatch "${COMMON[@]}" --job-name="sig-$stage" --time="$t" ${dargs[@]+"${dargs[@]}"} "$@" cluster/run_signals.sbatch
}
if [[ "$MODE" == "gate" ]]; then
  j=$(sub gate "$T_GATE" "")
  echo "submitted gate=$j  (watch: squeue -u \$USER;  result: Slack + $RUN_DIR/signals_gate/gate_report_eval.txt)"
  notify "signals gate SUBMITTED" "gate=$j"
  exit 0
fi
ck=$(sub ckpt "$T_CKPT" "" "${LIGHT[@]}")
sc=$(sub score "$T_SCORE" "$ck")
st=$(sub states "$T_STATES" "$ck" "${LIGHT[@]}")
msg="ckpt=$ck score=$sc states=$st"
dep="$sc"
if [[ "$SIG_ROLLOUTS" == "1" ]]; then
  ro=$(sub rollouts "$T_ROLL" "$sc"); dep="$sc:$ro"; msg="$msg rollouts=$ro"
fi
de=$(sub derive "$T_DERIVE" "$dep" "${LIGHT[@]}")
pr=$(sub probes "$T_PROBE" "$de:$st" "${LIGHT[@]}")
an=$(sub analyze "$T_AN" "$pr" "${LIGHT[@]}")
msg="$msg derive=$de probes=$pr analyze=$an"
echo "submitted: $msg   (watch: squeue -u \$USER)"
notify "signals pipeline SUBMITTED" "$msg"
