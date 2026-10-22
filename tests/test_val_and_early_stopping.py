"""Validation split, per-head val MAE accumulation, and the ValPlateauController."""
import types

import torch
import torch.nn as nn

from miol.data.hidden_states_dataset import HiddenStatesDataset, split_indices
from miol.model.callback import ValPlateauController
from miol.model.offline_trainer import MultiProbeOfflineTrainer


def test_split_is_deterministic_disjoint_and_complete():
    tr, va = split_indices(1000, 0.05, 0)
    assert len(va) == 50 and len(tr) == 950
    assert set(tr).isdisjoint(va) and sorted(tr + va) == list(range(1000))
    assert (tr, va) == split_indices(1000, 0.05, 0)
    assert va != split_indices(1000, 0.05, 1)[1]


def test_split_never_empties_either_side():
    tr, va = split_indices(3, 0.0001, 0)
    assert len(va) == 1 and len(tr) == 2
    tr, va = split_indices(3, 0.99, 0)
    assert len(tr) == 1


def test_dataset_subset_keeps_mode_and_files(tmp_path):
    d = HiddenStatesDataset.__new__(HiddenStatesDataset)
    d.sample_files = [tmp_path / f"s{i}.safetensors" for i in range(10)]
    d.layers_to_train, d.hidden_layer = [0, 1], "all"
    sub = d.subset([2, 5, 7])
    assert len(sub) == 3 and sub.layers_to_train == [0, 1] and sub.hidden_layer == "all"
    assert sub.sample_files == [d.sample_files[i] for i in (2, 5, 7)]
    assert len(d) == 10  # original untouched


def test_accumulate_and_consume_val_mae():
    t = types.SimpleNamespace(_val_ae={}, _count_probe_names=["count", "baseline_count"])
    acc = MultiProbeOfflineTrainer._accumulate_val_ae
    preds = torch.tensor([[10.0, 20.0, 99.0]])
    targets = torch.tensor([[8, 25, -100]])          # last token masked
    acc(t, "count", preds, targets)
    acc(t, "count", torch.tensor([[1.0]]), torch.tensor([[4]]))
    out = MultiProbeOfflineTrainer.consume_val_mae(t)
    assert abs(out["count"] - (2 + 5 + 3) / 3) < 1e-9
    assert "baseline_count" not in out and t._val_ae == {}


class _Sched:
    def __init__(self): self.stepped = []
    def step(self, v): self.stepped.append(v)


def _fake_trainer(script, tmp_path):
    heads = nn.ModuleDict({"count_L0": nn.Linear(2, 1), "count_L1": nn.Linear(2, 1), "cls_5": nn.Linear(2, 5)})
    it = iter(script)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(heads=heads),
        probe_specs={"count_L0": {"base_probe_name": "count"}, "count_L1": {"base_probe_name": "count"},
                     "cls_5": {}},
        _extra_schedulers={"count_L1": _Sched()},
        consume_val_mae=lambda: next(it),
    ), heads


def _run(ctrl, trainer, heads, evals, tmp_path):
    from transformers import TrainerControl
    args = types.SimpleNamespace(output_dir=str(tmp_path))
    stops = []
    for i in range(evals):
        heads["count_L0"].weight.data.fill_(float(i))        # distinct weights per eval
        heads["cls_5"].weight.data.fill_(float(i))
        ctl = TrainerControl()
        ctrl.on_evaluate(args, types.SimpleNamespace(global_step=(i + 1) * 100), ctl)
        stops.append(ctl.should_training_stop)
    ctrl.on_train_end(args, types.SimpleNamespace(global_step=evals * 100), TrainerControl())
    return stops


def test_early_stop_after_patience_and_restore_best(tmp_path):
    # monitored = mean(L0, L1): 50, 40 (best @ eval 2), then worse
    script = [{"count_L0": 60.0, "count_L1": 40.0}, {"count_L0": 45.0, "count_L1": 35.0}] + \
             [{"count_L0": 50.0, "count_L1": 36.0}] * 5
    trainer, heads = _fake_trainer(script, tmp_path)
    ctrl = ValPlateauController(patience=3, min_delta=0.25)
    ctrl.trainer = trainer
    stops = _run(ctrl, trainer, heads, 5, tmp_path)
    assert stops == [False, False, False, False, True]
    assert ctrl.best_step == 200
    # best snapshot was taken when weights were filled with 1.0 (second eval)
    assert torch.all(heads["count_L0"].weight == 1.0)
    assert torch.all(heads["cls_5"].weight == 1.0)       # non-val head follows best step
    assert len(trainer._extra_schedulers["count_L1"].stepped) == 5
    assert (tmp_path / "early_stopping.json").exists()


def test_min_delta_ignores_noise_and_improvement_resets_counter(tmp_path):
    script = [{"count_L0": 50.0, "count_L1": 50.0}, {"count_L0": 49.9, "count_L1": 49.9},
              {"count_L0": 40.0, "count_L1": 40.0}, {"count_L0": 39.9, "count_L1": 39.9}]
    trainer, heads = _fake_trainer(script, tmp_path)
    ctrl = ValPlateauController(patience=2, min_delta=0.25)
    ctrl.trainer = trainer
    stops = _run(ctrl, trainer, heads, 4, tmp_path)
    assert stops == [False, False, False, False]          # bad_evals: 0,1,0,1 -> never reaches 2
    assert ctrl.best_step == 300


def test_per_head_best_state_for_sweep(tmp_path):
    script = [{"count_L0": 10.0, "count_L1": 90.0}, {"count_L0": 20.0, "count_L1": 30.0}]
    trainer, heads = _fake_trainer(script, tmp_path)
    ctrl = ValPlateauController(patience=9, min_delta=0.0)
    ctrl.trainer = trainer
    heads["count_L1"].weight.data.fill_(0.0)
    from transformers import TrainerControl
    args = types.SimpleNamespace(output_dir=str(tmp_path))
    ctrl.on_evaluate(args, types.SimpleNamespace(global_step=100), TrainerControl())
    heads["count_L0"].weight.data.fill_(7.0); heads["count_L1"].weight.data.fill_(5.0)
    ctrl.on_evaluate(args, types.SimpleNamespace(global_step=200), TrainerControl())
    heads["count_L0"].weight.data.fill_(9.0); heads["count_L1"].weight.data.fill_(9.0)
    ctrl.on_train_end(args, types.SimpleNamespace(global_step=200), TrainerControl())
    assert ctrl.best_head_step == {"count_L0": 100, "count_L1": 200}
    assert torch.all(heads["count_L1"].weight == 5.0)     # L1 restored from step 200
    assert not torch.all(heads["count_L0"].weight == 9.0) # L0 restored from step 100
