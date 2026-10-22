#!/bin/bash
# Submit the whole pipeline as a dependent chain. Run from the repo root on the login node:
#   SMOKE=1 bash cluster/submit.sh      # tiny rehearsal first (minutes)
#   bash cluster/submit.sh              # the real run
# Order:  extract -> seed 0 (train, eval, layer sweep) -> seed 1 -> seed 2 -> final combine.
# After each seed finishes you get its results on Slack, so you can work with seed 0 while 1 and 2 run.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source cluster/config.sh
[[ -z "$SBATCH_ACCOUNT" ]] && { echo "Set SBATCH_ACCOUNT in cluster/config.sh"; exit 1; }
export LAYER_SWEEP="${LAYER_SWEEP:-1}"     # 1 = per-layer sweep too (default); LAYER_SWEEP=0 skips it

COMMON=(--parsable --account="$SBATCH_ACCOUNT" --export=ALL)
[[ -n "$SLACK_EMAIL" ]] && COMMON+=(--mail-user="$SLACK_EMAIL" --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80)
if [[ "$SMOKE" == "1" ]]; then T_EXTRACT=00:40:00 T_SEED=01:30:00 T_FINAL=00:30:00
else T_EXTRACT=04:00:00 T_SEED=12:00:00 T_FINAL=02:00:00; fi

prev=$(STAGE=extract sbatch "${COMMON[@]}" --job-name=mop-extract --time="$T_EXTRACT" cluster/run.sbatch)
msg="extract=$prev"
for seed in $SEEDS; do
  prev=$(STAGE=seed SEED="$seed" sbatch "${COMMON[@]}" --job-name="mop-seed$seed" --time="$T_SEED" --dependency=afterok:"$prev" cluster/run.sbatch)
  msg="$msg seed$seed=$prev"
done
final=$(STAGE=final sbatch "${COMMON[@]}" --job-name=mop-final --time="$T_FINAL" --dependency=afterok:"$prev" --gpus-per-node=1 --cpus-per-task=8 cluster/run.sbatch)
msg="$msg final=$final"
echo "submitted: $msg   (watch: squeue -u \$USER)"
notify "pipeline SUBMITTED" "$msg"
