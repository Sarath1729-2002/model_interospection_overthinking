# Running on Compute Canada (Alliance) - from the login node

Everything below is typed on the **login node** (`ssh you@<cluster>.alliancecan.ca`). You never run the heavy
work yourself: `submit.sh` queues a chain of SLURM jobs: extract once, then one job per seed (train, evaluate, layer sweep) in order, then a final job that combines the seeds, on 4 GPUs each (data parallel).
Slack reports arrive by e-mail-to-Slack at each stage.

## 0. Get the code onto the cluster
```bash
git clone <your model_overthinking_probe repo> && cd model_overthinking_probe
# (or copy the tarball:  tar xzf model_overthinking_probe.tar.gz && cd model_overthinking_probe)
```

## 1. Fill in two values in `cluster/config.sh`
* `SLACK_EMAIL`: your personal Slack forwarding address, which delivers to your DM with Slackbot. Slack: profile picture -> Preferences -> Messages & media -> "Bring emails into Slack" -> Get a Forwarding Address. (A channel address is paid-plan only and may be admin-restricted.)
* `SBATCH_ACCOUNT`: your allocation, e.g. `def-<pi>`. List yours: `sacctmgr show assoc user=$USER format=account%30`.

## 2. One-time setup (needs internet, so login node only, ~10-20 min)
```bash
bash cluster/setup.sh
```
Makes the Python env, downloads Qwen2.5-7B-Instruct + GSM8K into `$SCRATCH/model_overthinking_probe/hf_home`
(compute nodes are offline), and prints the tokenizer/EOS/generation_config it found. Read that output once.

## 3. Test your Slack address (10 seconds)
```bash
source cluster/config.sh && notify "hello from the cluster" "test"
```
If nothing shows up in Slack: the cluster may have no `mail` command (then you only get the SLURM BEGIN/END/FAIL
mails, if any, and the log lines), or Slack rejects that sender. Try a personal address first to separate the two.

## 4. Rehearse (minutes), then run
```bash
SMOKE=1 bash cluster/submit.sh      # 64 train / 32 eval prompts, 40 steps - checks the whole chain
bash cluster/submit.sh              # the full run: extract -> seed 0 -> seed 1 -> seed 2 -> final combine (seed 0 results reach Slack before seeds 1-2 start)
LAYER_SWEEP=0 bash cluster/submit.sh   # same without the layer sweep
```
`submit.sh` prints three job ids. Watch with `squeue -u $USER`; logs are `mop-<stage>-<jobid>.out (`ls mop-*`)` in the repo folder
(`tail -f mop-<stage>-<jobid>.out (`ls mop-*`)`). If a job dies or times out, fix/extend and run `submit.sh` again: extraction resumes
where it stopped (finished samples are skipped).

In the rehearsal's extraction log, look at `samples/s` and `generated tokens/s` per rank and extrapolate the full time.

## 5. What you get on Slack (your DM)
`pipeline SUBMITTED` -> `extraction STARTED/DONE` (truncation rate, token lengths, disk used) ->
**seed 0:** `training STARTED` -> `main results ready` (MAE of every probe and baseline) -> `layer sweep STARTED` -> `layer curve ready` (MAE per layer, best layer) ->
seed 1 and seed 2 repeat the same four messages -> `ALL SEEDS DONE - final results` (all-seed summary + layer curve with seed error bars).
Any failure sends `FAILED: ...` with the last 25 log lines. SLURM also mails BEGIN/END/FAIL/80%-of-time to the same address.
Plots are not attached to the message (plain-text mail); they are on disk (see 6).

## 6. After it finishes
Results: `$SCRATCH/model_overthinking_probe/outputs/qwen2.5_7b-gsm8k/` (symlinked as `outputs/` in the repo):
`hidden_states/` (cache + `audit.txt`), `summary.txt`, and per seed `qwen2.5_7b-gsm8k-<seed>/seed_<seed>/{trained_probe,eval_results.json}`.
Per seed: `summary_seed<N>.txt`, `layer_sweep_seed<N>/`. Final: `summary.txt` and `layer_sweep/{layer_sweep.png,layer_sweep.csv,layer_sweep.json,summary.txt}` (per-layer MAE with 95% CI and seed std).
W&B ran offline; upload from the login node: `wandb sync $SCRATCH/model_overthinking_probe/wandb/wandb/offline-run-*`.

## Things to check, not assume
* **Disk**: estimate ~0.6 TB for the full GSM8K cache (unverified; the audit prints real token counts). Check `diskusage_report`
  before launching; scratch is purged after ~60 days of inactivity.
* **Settings I chose** (all in `cluster/config.sh`): `MAX_NEW_TOKENS=1024` (repo default 300 would truncate GSM8K answers),
  `repetition_penalty=1.0` (Qwen's checkpoint ships 1.05), `EVAL_STEPS=100` (each eval re-reads the validation samples from disk),
  3 seeds. Exact repo cadence: `EVAL_STEPS=10 SCHED_PATIENCE=5 ES_ENABLED=0 VAL_FRACTION=0 bash cluster/submit.sh`.
* **Validation split + early stopping**: `VAL_FRACTION=0.05` of TRAIN is held out (same samples for every seed). The LR schedule,
  early stopping (`ES_PATIENCE=5` evals without `ES_MIN_DELTA=0.25` tokens of improvement) and best-step selection use only that
  split; each count head is restored to its own best validation step; the test split is scored once, by `evaluate_probes.py`.
  `MAX_STEPS` is now only a cap. `early_stopping.json` in each `trained_probe/` folder records the best/stop steps.
  Every count head (not just the first) now decays its own learning rate on its own validation MAE.
* **Ridge cross-check** (`RIDGE=1`, own job `mop-ridge`, starts after extraction): closed-form ridge per layer, strength picked on the
  validation split, test MAE with 95% CI next to constant/position baselines -> `$RUN_DIR/ridge_check/summary.txt`. If ridge is clearly
  below the trained probe at the same layer, the trained probe is optimisation-limited. When re-submitting only extra seeds use `RIDGE=0`.
* **GPUs/memory**: `GPUS_PER_NODE` (default `h100:4`, TamIA gives whole nodes), `MEM` (default `0`; Fir: `MEM=240G`), `FINAL_GPUS` for the light jobs.
* Not tested on a real cluster or with the real Qwen weights (this was built where those were unreachable); the rehearsal is the test.
