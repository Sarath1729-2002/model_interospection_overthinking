# MATH signals run (Qwen2.5-7B-Instruct, integer-answer MATH)

Everything is regenerated from scratch: one pass per chunk of problems generates the solution, caches the checkpoint hidden states
(29 layers), token entropy/surprisal and the forced-answer scores.  No cached GSM8K extraction is needed.

## Order (cluster folder = `model_interospection_overthinking`, never edit on the cluster, do not pull while jobs run)
1. Login node, once:           `bash cluster/setup_math.sh`          (downloads MATH, writes problems_{train,test}.jsonl)
2. Gate (~30 min):             `MODE=gate bash cluster/submit_math.sh`   -> read `$RUN_DIR/signals_gate/gate_report_eval.txt`
3. Full:                       `MODE=full bash cluster/submit_math.sh`
   (adds resampled rollouts with `MSIG_ROLLOUTS=1`; skip them first, they were weak on GSM8K and cost 1.5-2.5 h)
4. After the full run:         `MODE=extras bash cluster/submit_math.sh`  (only if transfer/steer were not chained)
Run dir: `$ROOT/outputs/qwen2.5_7b-math/signals`. Read `analysis/report.md`, `analysis/controls.md`, `transfer.md`, `steer.md`.
Resubmitting is safe: finished chunks are skipped.  Do not change MSIG_CHUNK / MSIG_*_N between a run and its resume.

## Time (estimates, not measured on MATH)
worker ~1-1.5 h on 4 GPUs (generation-bound, max_new 2048) | merge 10-20 min | probes 40-60 min | controls ~20 min | analyze ~10 min | transfer/steer ~10 min each. Storage ~20 GB.

## What is new vs GSM8K
Forecast targets (a_next, da, loss_next, gain_next, loss_end, y_end, ent_next, surp_next, hedge_next, flips_rem, steps_rem, len_next),
logit-statistics baselines B5/B5_mlp (whole-prefix bag of words + entropy features), `_core` and wobble conditions, AUROC table of
events, Q0 control, GSM8K->MATH transfer, steering with random-direction control.

## Caveats
- Only integer-answer problems are used (deterministic gold). That biases toward algebra/number theory/counting.
- Traces hitting the 2048 cap are flagged truncated in the gate report; they are kept but have no final answer.
- Steering is blunt (additive at all positions); a null does not rule out a causal role. Pass negative alphas as `--alphas=-0.3,0.3`.
- Novelty is unverified; the idea is crowded (early-exit probes, hidden error awareness, ...).
