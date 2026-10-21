"""Print mean MAE per probe/baseline (mean over eval examples) for each seed. Used for the Slack report.

    python summarize_eval.py <run_dir> [seed]     # no seed = all seeds found
"""
import json, sys
from pathlib import Path

SEED = sys.argv[2] if len(sys.argv) > 2 else "*"
PATTERN = f"*/seed_{SEED}/eval_results.json"

def scalar(v):
    if isinstance(v, dict):
        v = v.get("mean")
    return v

for f in sorted(Path(sys.argv[1]).glob(PATTERN)):
    data = json.load(open(f))
    acc = {}
    for ex in data["examples"]:
        for name, entry in (ex.get("probes") or {}).items():
            agg = (entry or {}).get("aggregate") or {}
            m = scalar(agg.get("mae"))
            if m is not None:
                acc.setdefault(name, []).append(m)
        for name, entry in (ex.get("statistical_baselines") or {}).items():
            m = scalar(((entry or {}).get("aggregate") or {}).get("mae"))
            if m is not None:
                acc.setdefault("baseline:" + name, []).append(m)
    print(f"{f.parent.name}  ({len(data['examples'])} eval examples)  mean token-level MAE:")
    for k, v in sorted(acc.items()):
        print(f"   {k:<40s} {sum(v) / len(v):9.2f}")

# all-layers-concat reference line for the layer-sweep plot: mean over seeds of the per-completion MAE of `count`
vals = []
for f in sorted(Path(sys.argv[1]).glob(PATTERN)):
    if "layers" in f.parent.parent.name:
        continue
    ms = [scalar((((ex.get("probes") or {}).get("count") or {}).get("aggregate") or {}).get("mae")) for ex in json.load(open(f))["examples"]]
    ms = [m for m in ms if m is not None]
    if ms:
        vals.append(sum(ms) / len(ms))
if vals:
    Path(sys.argv[1], "all_layers_mae.txt").write_text(f"{sum(vals) / len(vals):.6f}")
