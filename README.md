# model_overthinking_probe

Linear probes on a frozen LLM's hidden states that predict how many tokens of its own answer are still to come.
This project extends the code of *How Much is Left? LLMs Linearly Encode Their Remaining Output Length*
(Merzouk et al.; upstream repo `model_introspection_output_length`, forked from commit `f9600dc`, MIT licence kept in `LICENSE`)
to **Qwen2.5-7B-Instruct on the full GSM8K**, with batched/async hidden-state extraction and a small Alliance (Compute Canada) runner.

* Running on the cluster: **[cluster/GUIDE.md](cluster/GUIDE.md)**
* Python package is still called `miol` (kept so the upstream code and configs work unchanged).
* Tests: `pip install -e ".[dev]" && pytest` (164 tests, CPU only).

---
Upstream README follows.

# How Much is Left? LLMs Linearly Encode Their Remaining Output Length

Code for a study of whether the residual stream of a frozen LLM linearly encodes an internal estimate of how many tokens of its own response are still to come. Linear probes are trained on the hidden states of three open-weight 7--8B instruction-tuned models across seven completion-style datasets; the probe (i) decodes total response length from the prompt's last hidden state, (ii) transfers across datasets, and (iii) spikes upward at retraction tokens.

## Setup

Python ≥ 3.12. Install with [uv](https://github.com/astral-sh/uv) (or `pip install -e .`):

```bash
uv sync
```

Configuration is driven by [Hydra](https://hydra.cc); the central config is [src/miol/conf/config.yaml](src/miol/conf/config.yaml). Every CLI invocation can override any field, e.g. `model=llama3.1_8b data=gsm8k seed=1`.

## Workflow

The pipeline runs in two stages: extract once, train many times. The base model is frozen throughout.

### 1. Extract hidden states

A single forward pass over `(prompt, completion)` pairs writes one `.safetensors` (per-layer hidden states) plus one `.json` (metadata) per sample, under `${output_dir}/hidden_states/{train,eval}/`.

```bash
# Single GPU
python src/miol/extract_hidden_states.py \
    model=llama3.1_8b \
    data=gsm8k \
    output_dir=outputs/llama3.1_8b-gsm8k

# Multi-GPU (one rank per GPU; samples sharded across ranks)
python -m torch.distributed.run --nproc_per_node=gpu \
    src/miol/extract_hidden_states.py \
    model=llama3.1_8b data=gsm8k \
    output_dir=outputs/llama3.1_8b-gsm8k
```

Truncated samples (`has_eos=False`) are dropped from the saved tensors but kept in metadata for auditing; toggle with `extract.save_truncated=true`.

### 2. Train the probe family offline

`train_offline.py` consumes the saved hidden states and trains *all* probe heads simultaneously in one pass. The default head set ([src/miol/conf/config.yaml](src/miol/conf/config.yaml#L65-L76)) covers the families described in §3.4 of the paper:

| Probe key             | Type           | Reads               | Paper name                          |
|---                    |---             |---                  |---                                  |
| `count`               | regression     | $h_t$               | Remaining Count Probe               |
| `percentage`          | regression     | $h_t$               | (auxiliary; normalized progress)    |
| `cls_K` (K=2,3,5,7,9) | classification | $h_t$               | Complementary classification probes |
| `prompt_only_count`   | regression     | $h_{\text{prompt-end}}$ | Completion Length Probe (countdown) |
| `baseline_count`      | constant       | —                   | Statistical baseline (median)       |
| `baseline_position_count` | position-only | $t$              | Length-minus-position floor         |

```bash
python -m torch.distributed.run --nproc_per_node=gpu \
    src/miol/train_offline.py \
    model=llama3.1_8b \
    data=hidden_states \
    data.hidden_states_dir=outputs/llama3.1_8b-gsm8k/hidden_states \
    data.dataset_name=gsm8k \
    output_dir=outputs/llama3.1_8b-gsm8k \
    seed=0 max_steps=4000 unified_batch_size=8 \
    data.hidden_layer=all
```

`output_dir` is auto-suffixed with `seed_${seed}` so multi-seed runs don't collide. The trained heads land in `${output_dir}/seed_${seed}/trained_probe/`.

#### Per-layer probe sweep

To duplicate every $h_t$-reading probe across a list of layers in a single run (the per-layer sweep reported in the paper's appendix):

```bash
python src/miol/train_offline.py \
    data=hidden_states \
    data.hidden_states_dir=... \
    'data.layers_to_train=[0,8,16,24,32,-1]'   # explicit list, or "all"
```

Heads are namespaced as `count_L0`, `count_L8`, ...; W&B logs `count_L*/eval_mae` per layer.

### 3. Evaluate / inspect

`evaluate_probes.py` re-applies every saved head plus every statistical baseline to the eval split and writes a single JSON suitable for the static dashboard at [src/miol/dashboard/](src/miol/dashboard/):

```bash
python src/miol/evaluate_probes.py \
    --probe-dir outputs/llama3.1_8b-gsm8k/seed_0/trained_probe \
    --hidden-states-dir outputs/llama3.1_8b-gsm8k/hidden_states \
    --output outputs/llama3.1_8b-gsm8k/seed_0/eval_results.json
```

For an interactive prompt → generation → per-token probe trace (Figure 2 in the paper):

```bash
python src/miol/interactive_probe_inference.py \
    --probe-dir outputs/llama3.1_8b-gsm8k/seed_0/trained_probe \
    --model-name meta-llama/Llama-3.1-8B-Instruct
```

### 4. Cross-dataset evaluation (paper §5)

`analysis/cross_dataset_eval.py` takes a fixed model and a list of datasets and fills in every (train, eval, seed) cell of the cross-dataset matrix. It expects probe heads under `outputs/{model}-{dataset}/seed_{seed}/trained_probe/` and hidden states under `outputs/{model}-{dataset}/hidden_states/` (both produced by stages 1 and 2 above):

```bash
python -m miol.analysis.cross_dataset_eval \
    --model llama3.1_8b \
    --datasets gsm8k math mmlu_pro openthoughts-1k trivia_qa count countdown \
    --seeds 0 1 2 \
    --probe count --metric mae \
    --output figures/cross_dataset/llama3.1_8b_count_mae.pdf
```

## Models and datasets

Model configs live in [src/miol/conf/model/](src/miol/conf/model/); the three reported in the paper are `llama3.1_8b`, `olmo3_7b`, `mistral_7b`. Dataset configs live in [src/miol/conf/data/](src/miol/conf/data/); the seven used in the paper are `gsm8k`, `math`, `mmlu_pro`, `openthoughts-1k`, `trivia_qa`, plus the synthetic `count` / `countdown` variants of [synthetic.yaml](src/miol/conf/data/synthetic.yaml).

## Repo layout

```
src/miol/
    extract_hidden_states.py      # stage 1
    train_offline.py              # stage 2 (multi-probe + multi-layer)
    evaluate_probes.py            # stage 3 (offline eval -> dashboard JSON)
    interactive_probe_inference.py
    statistical_baseline.py       # constant / position baselines
    analysis/
        cross_dataset_eval.py
    model/
        offline_trainer.py        # MultiProbeOfflineTrainer
        probe_specs.py            # registry of probe heads
        ...
    conf/                         # Hydra configs
    dashboard/                    # static HTML/JS dashboard
tests/                             # pytest suite for generation utilities
```

## Reproducing the paper's tables

Each headline cell in the paper is the mean across three independent training seeds. For one (model, dataset) pair, the recipe is: extract hidden states once, then run stage 2 three times with `seed=0`, `seed=1`, `seed=2`. The cross-dataset matrix (paper Tables 3–5) is then produced by stage 4 once per `(model, probe, metric)` tuple, pointing at the resulting `outputs/{model}-{dataset}/` tree.
