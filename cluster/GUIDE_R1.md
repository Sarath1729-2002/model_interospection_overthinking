# DeepSeek-R1-Distill-Qwen-7B vs Qwen2.5-7B-Instruct on MATH (one-on-one)

Same pipeline, same cue, same probes, same held-out problems; only the model changes.  R1-Distill-Qwen-7B has the same architecture
as Qwen2.5-7B (28 layers, 3584 dims), so nothing in the probe stages changes.  Differences handled by `MSIG_STYLE=r1`:
no system prompt (the same instruction sits in the user turn), prompt ends inside an open `<think>` block, sampling T=0.6 / top_p=0.95
(DeepSeek's recommendation), `MSIG_MAX_NEW=6144`, generation batches of 32, at most 40 checkpoints per trace (even spread; long
traces would otherwise have 100+), train problems 2000 (set `MSIG_TRAIN_N` to change), transfer stage skipped (no GSM8K states for R1).

## Order  (cluster folder `model_interospection_overthinking`; never edit on the cluster; do not pull while jobs run)
0. The Qwen MATH run must be finished (its problem lists are copied so both models see the SAME problems).
1. Login node, once:    `bash cluster/setup_r1.sh`          (downloads the model, copies the problem lists, prints the chat template)
   -> check the printed template ends with `<think>` (the code appends it if not) and look at the eos token.
2. Gate (~45 min):      `MODE=gate bash cluster/submit_r1.sh`     -> `$ROOT/outputs/r1_distill_7b-math/signals_gate/gate_report_eval.txt`
   Read: truncation % (R1 may loop to the cap), accuracy, G1 (cue works: median P(gold) at the end of correct traces >= 0.9).
   If G1 fails, the default cue does not work on R1: rerun the gate with `MSIG_CUE='Wait, I suddenly got the answer to the whole problem. **Final Answer**\n\n\[\n\boxed{'`
   (a cue style used in the early-exit literature; it makes the two runs less comparable, so prefer the default if it passes).
3. Full:                `MODE=full bash cluster/submit_r1.sh`
4. Compare (login node, CPU, seconds):
```bash
cd ~/model_interospection_overthinking
unset RUN_NAME RUN_DIR; export DATA=math; source cluster/config.sh        # sets ROOT
python cluster/msig_compare.py --a $ROOT/outputs/qwen2.5_7b-math/signals --b $ROOT/outputs/r1_distill_7b-math/signals --name-a qwen --name-b r1
```
Writes `r1_distill_7b-math/signals/analysis/compare_qwen_vs_r1.md`.  Also read that run's own `analysis/report.md`, `controls.md`, `steer.md`.

## Time (estimates, not measured)
Generation dominates: 32 sequences x up to 6144 steps ~ 3-4 min per batch, x ~27 batches per GPU for 3000 problems on 4 GPUs ~ 1.5-2 h.
Scoring: 40 forced-answer contexts of ~2-4k tokens per trace ~ 1 h.  Worker total ~ 3-4 h (job limit 24 h).  Merge ~20 min, probes ~1 h,
controls/analyze/steer ~30 min.  Storage: ~200 KB per checkpoint -> ~25 GB for chunks + ~25 GB merged states.  Delete `chunks/` after the merge if space is short.

## Caveats
- Truncated traces (no </think> + answer within the cap) are kept but have no final answer and no checkpoints; report the truncation rate when comparing accuracy.
- R1 traces often contain intermediate `\boxed{}`; the parser takes the last one, which is the final answer when the trace finishes.
- Checkpoints are capped at 40 per trace, so `steps_rem`, `hedge_next` and `len_next` span several paragraphs for long traces; compare those targets with care.
- Everything about novelty and publishability from the Qwen run still applies; this run tests whether the signals get stronger when overthinking is more extreme.
