# Signals experiment: linear probes for what the model *knows* and is *committed to* along its reasoning

Built on the cached Qwen2.5-7B-Instruct / GSM8K run (`$RUN_DIR/hidden_states`, from `cluster/submit.sh`). Nothing is re-generated:
we cut each cached completion at its blank-line boundaries and ask the model, at every cut, what it would answer *right now*.

## What is measured (every label is exact; no judge)
At each checkpoint the prefix is followed by the cue `Therefore, the final answer is \(\boxed{` and we read the model's answer distribution:

| target | meaning |
|---|---|
| `a` | log P(gold answer) - what the model **knows** here |
| `f` | log P(the trace's own final answer) - what it is **committed** to (own answer parsed from the completion) |
| `g = f - a` | committed against the truth |
| `fg`, `ff` | greedy forced answer == gold / == own final answer |
| `d_gold`, `d_commit` | tokens from here to the stable-convergence / commitment point (negative = not there yet) |
| `rem`, `pct` | the paper's length target (tokens left, fraction left) on the same checkpoints |
| `r_pgold`, `r_pfinal`, `r_H` | (rollouts) fraction of M resampled continuations ending on gold / own answer, and their answer entropy |

Derived per trace: `t*` (forced answer is gold from here on), `c*` (forced answer equals the trace's own answer from here on), slack
(tokens still written after), flips, and four named conditions (threshold `SIG_TAU` tokens):
**overthink** (correct, converged, kept writing), **rationalize** (wrong, committed, kept writing), **lost** (wrong trace that had the gold answer at some step), **dither** (forced answer changed >= 2 times).
The mapping of these onto a safety story (overthinking, rationalisation, ...) is deliberately *not* baked in; the probes come first.

Probes: closed-form **ridge regressions** on the frozen residual stream (per layer, and all 29 layers concatenated via exact dual ridge),
lambda chosen on the validation split (the same fixed 5% of train problems as the paper's probes). Test = the eval problems.
Baselines B0-B4 block the boring explanations: B1 position, B2 + prompt length, B3 + "is the answer already written" flags (ridge and a small MLP),
B4 + hashed bag-of-words of the last ~200 characters. Uncertainty = bootstrap over problems.

## Run it (login node, repo root; `cluster/config.sh` already filled in for the main run)
```bash
# 0. cluster/submit.sh must have finished extraction (this reads $RUN_DIR/hidden_states). Do not pull while jobs run.
SMOKE=1 MODE=full bash cluster/submit_signals.sh     # optional rehearsal on the -smoke run (run SMOKE=1 bash cluster/submit.sh first)
MODE=gate bash cluster/submit_signals.sh              # 1. feasibility gate, ~15-25 min.  READ IT BEFORE GOING ON
MODE=full bash cluster/submit_signals.sh              # 2. everything: ckpt -> score || states -> [rollouts] -> derive -> probes -> analyze
SIG_ROLLOUTS=0 MODE=full bash cluster/submit_signals.sh   # same without the resampling arm (saves ~1-1.5 h)
```
The gate (`$RUN_DIR/signals_gate/gate_report_eval.txt`, also on Slack) checks on 300 eval traces:
G1 the cue works (median P(gold) >= 0.9 at the final checkpoint of correct traces; gold scorable on >= 95%),
G2 `a` moves (>= 50% of traces rise >= 3 nats from the prompt end to the end),
G3 some correct traces converge before the end (>= 20%).
If G1 fails, fix the cue (`--cue` in `sig_score.py`) before anything else; if G2/G3 fail the signal is not there for this model/dataset and the full run will not rescue it.
Stages are resumable (finished chunks are skipped); resubmit the same command after a timeout.

## Outputs (`$RUN_DIR/signals/`)
`ckpt_*.csv`, `traces_*.csv` (tables) - `score_parts/`, `roll_parts/` (raw chunks) - `labels_*.csv`, `traces_labeled_*.csv`, `gate_report_*.txt`
- `states_*.npy` (float16 [layers, checkpoints, 3584]; ~16 GB train) - `probes/{preds.npz,meta.json,scores_test.csv}` -
**`analysis/report.md`** (+ csv per table, `layer_curves.png`, `trajectories.png`).

`report.md` sections: headline (probe vs B4), condition prevalence, best layer per target, paired probe-vs-baseline tables, within-trace ordering,
"before the text shows it" subsets, condition monitors, early-exit simulation, rollout validation.

## Time estimate (8.8k traces, ~10 checkpoints each; NOT measured on the cluster - the gate gives the real scoring rate)
| stage | est. | notes |
|---|---|---|
| ckpt | 5-15 min | CPU, decodes each prefix once |
| score | 25-50 min | 4 GPUs; ~85k contexts x ~300 tokens, two forward passes (answer log-prob, greedy) |
| states | 30-90 min | I/O bound: reads the cached tensors once (~0.6 TB); runs next to score |
| rollouts | 1-1.5 h | 600 eval + 2000 train traces x 4 continuations; HF generate, no vLLM |
| derive | < 2 min | |
| probes | 30-60 min | 29 per-layer ridge fits (a few eigh each) + the all-layer dual ridge (50k rows) + baselines |
| analyze | 5-10 min | |
Wall clock with rollouts roughly 3-4 h, without 2-2.5 h, plus queue time.

## Caveats (read before believing a number)
* Nothing here was run on the real model: the code is tested on a tiny random Qwen2 + synthetic cache (tests/test_signals.py). The rehearsal and the gate are the real test.
* The cue is a choice. `a` measures what the model would say *when forced in this format*, which is not necessarily what it would say if allowed to continue. Rollouts (`r_*`) are the check; read `rollout_validation` first.
* The PoC lesson stands: most of a probe's score can come from position or from the answer already being visible in the text. Compare to **B4**, use the within-trace and "before the text shows it" tables, not the pooled R2.
* `d_gold` / `d_commit` exist only for traces that converge / commit, so those targets are fit on a selected subset.
* The all-layer probe is fit on a random subset of at most `SIG_KERNEL_MAX_FIT` (50k) checkpoints (memory); per-layer probes use all of them.
* Qwen2.5-7B on GSM8K may barely overthink; expect a modest overthink prevalence and say so if the gate shows it.
* Not built: steering / activation patching (the causal arm), trained (Adam) probes for the new targets, more seeds, MATH or other models.
