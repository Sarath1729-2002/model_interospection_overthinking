"""Sweep candidate batch sizes and dataloader worker counts.

Runs `train_offline.py` once per (batch_size, num_workers) combination with
`max_steps=N` and `wandb.enabled=false`, captures HuggingFace Trainer's
`train_runtime` / `train_samples_per_second` from stdout when available, and
prints a comparison table at the end.

Setup overhead (model load, dataset prep) is included in wall time but is
roughly constant across runs. For tighter numbers, increase `--steps` so setup
amortizes out.

Usage:
    # Sweep batch sizes only (workers from config):
    python src/miol/benchmark_batch_size.py \\
        --batch-sizes 4 8 16 32 \\
        --steps 50 \\
        -- \\
        data=gsm8k display_every_n_steps=0

    # Full grid of batch_size x num_workers:
    python src/miol/benchmark_batch_size.py \\
        --batch-sizes 8 16 \\
        --num-workers 2 4 8 16 \\
        --steps 100 \\
        -- \\
        data=hidden_states data.hidden_states_dir=outputs/.../hidden_states

Anything after the `--` is forwarded as Hydra overrides to `train_offline.py`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time


def _parse_train_runtime(output: str) -> dict[str, float] | None:
    """Pull HF Trainer's train_runtime / *_per_second metrics from stdout if present."""
    metrics = {}
    for key in ("train_runtime", "train_samples_per_second", "train_steps_per_second"):
        m = re.search(rf"['\"]{key}['\"]\s*[:=]\s*([0-9]+\.?[0-9]*)", output)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass
    return metrics if metrics else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple batch sizes by running train_offline.py for a fixed number of steps each.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="Candidate per-device batch sizes to sweep.")
    parser.add_argument("--num-workers", type=int, nargs="+", default=None,
                        help="Candidate dataloader.num_workers values. If omitted, uses whatever the config says.")
    parser.add_argument("--steps", type=int, default=50,
                        help="max_steps for each run. Larger values give better averaging but cost more time.")
    parser.add_argument("--script", default="src/miol/train_offline.py",
                        help="Path to the training entry point.")
    parser.add_argument("--config-name", default=None,
                        help="Hydra config name (e.g. config_debug).")
    parser.add_argument("hydra_args", nargs=argparse.REMAINDER,
                        help="Anything after `--` is forwarded as Hydra overrides.")
    args = parser.parse_args()

    # Strip the leading `--` if argparse left it in.
    extra = list(args.hydra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Grid of (batch_size, num_workers). If --num-workers wasn't provided, the
    # inner sweep contains a single None value meaning "leave the config's value".
    worker_values = args.num_workers if args.num_workers else [None]

    results: list[dict] = []
    aborted = False
    for bs in args.batch_sizes:
        if aborted:
            break
        for nw in worker_values:
            cmd = [sys.executable, args.script]
            if args.config_name:
                cmd.append(f"--config-name={args.config_name}")
            cmd += [
                f"max_steps={args.steps}",
                f"unified_batch_size={bs}",
                "wandb.enabled=false",
                "display_every_n_steps=0",
                "optimization.save_steps=1000000",  # don't save during benchmark
            ]
            if nw is not None:
                cmd.append(f"dataloader.num_workers={nw}")
                # prefetch_factor must be None when num_workers=0
                if nw == 0:
                    cmd.append("dataloader.prefetch_factor=null")
            cmd += extra

            print()
            print("=" * 70)
            label = f"batch_size={bs}, num_workers={nw if nw is not None else '(config)'}"
            print(f"  Running {label}, steps={args.steps}")
            print(f"  cmd: {' '.join(cmd)}")
            print("=" * 70)

            wall_start = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except KeyboardInterrupt:
                print("Interrupted.")
                aborted = True
                break
            wall_elapsed = time.time() - wall_start

            # Stream the captured output so the user sees progress / errors.
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)

            entry = {
                "batch_size": bs,
                "num_workers": nw,
                "wall_elapsed": wall_elapsed,
                "ok": proc.returncode == 0,
            }
            if proc.returncode != 0:
                print(f"  FAILED with exit code {proc.returncode}")
                results.append(entry)
                continue

            metrics = _parse_train_runtime(proc.stdout) or {}
            entry["train_runtime"] = metrics.get("train_runtime")
            entry["samples_per_second"] = metrics.get("train_samples_per_second")
            entry["steps_per_second"] = metrics.get("train_steps_per_second")
            results.append(entry)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    header = f"{'batch':>6} {'workers':>8} {'wall (s)':>10} {'train_rt (s)':>14} {'samples/s':>12} {'steps/s':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        nw = r.get("num_workers")
        nw_str = "(cfg)" if nw is None else str(nw)
        if not r["ok"]:
            print(f"{r['batch_size']:>6} {nw_str:>8} {'FAIL':>10}")
            continue
        wall = f"{r['wall_elapsed']:.1f}"
        rt = f"{r['train_runtime']:.1f}" if r.get("train_runtime") is not None else "—"
        sps = f"{r['samples_per_second']:.2f}" if r.get("samples_per_second") is not None else "—"
        spsec = f"{r['steps_per_second']:.3f}" if r.get("steps_per_second") is not None else "—"
        print(f"{r['batch_size']:>6} {nw_str:>8} {wall:>10} {rt:>14} {sps:>12} {spsec:>10}")
    print()
    print("Notes:")
    print("- 'wall (s)' includes startup (model load, dataset prep). Comparable across runs.")
    print("- 'train_rt' / 'samples/s' / 'steps/s' come from HF Trainer's final metrics dict.")
    print("- 'samples/s' here is per the global effective batch (per_device * num_gpus * grad_accum).")
    print("- For meaningful comparison use --steps 100 or higher so setup amortizes.")


if __name__ == "__main__":
    main()
