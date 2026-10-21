"""
Probe specifications for multi-probe training.

Each probe is a lightweight linear head that maps an input to predictions.
Regression probes output a single scalar; classification probes output a
probability distribution over bins.

input_type controls what is fed to the probe head:
- "hidden_states": the base model's hidden state at each position (default, main probes)
- "constant":      a vector of ones — probe learns a dataset-wide mean (no-input baseline)
- "position":      the position within the completion (tokens generated so far) —
                   a strong baseline for count prediction

train_position controls which sequence positions contribute to the loss:
- absent / "all":   every non-masked position contributes (default).
- "prompt_end":     only the prompt's last position contributes. The probe still
                    runs at every position at evaluation time, but the loss / target
                    tensor is -100 everywhere except at index `prompt_length - 1`.
"""

PROBE_SPECS = {
    "count": {
        "type": "regression",
        "input_type": "hidden_states",
        "output_dim": 1,
        "target_key": "remaining_token_counts",
    },
    "log_count": {
        # Trained on log(1 + remaining_token_count). Predictions are mapped back
        # to count space via expm1 before bucketing, so MAE/MSE/MAPE/SMAPE are
        # directly comparable to the `count` probe.
        "type": "regression",
        "input_type": "hidden_states",
        "output_dim": 1,
        "target_key": "remaining_log_count",
    },
    "percentage": {
        "type": "regression",
        "input_type": "hidden_states",
        "output_dim": 1,
        "target_key": "remaining_percentage",
    },
    "cls_2": {
        "type": "classification",
        "input_type": "hidden_states",
        "output_dim": 2,
        "target_key": "soft_targets_2",
    },
    "cls_3": {
        "type": "classification",
        "input_type": "hidden_states",
        "output_dim": 3,
        "target_key": "soft_targets_3",
    },
    "cls_5": {
        "type": "classification",
        "input_type": "hidden_states",
        "output_dim": 5,
        "target_key": "soft_targets_5",
    },
    "cls_7": {
        "type": "classification",
        "input_type": "hidden_states",
        "output_dim": 7,
        "target_key": "soft_targets_7",
    },
    "cls_9": {
        "type": "classification",
        "input_type": "hidden_states",
        "output_dim": 9,
        "target_key": "soft_targets_9",
    },
    # Baselines — used to establish performance floors for interpretation.
    "baseline_count": {
        "type": "regression",
        "input_type": "constant",
        "output_dim": 1,
        "target_key": "remaining_token_counts",
    },
    "baseline_percentage": {
        "type": "regression",
        "input_type": "constant",
        "output_dim": 1,
        "target_key": "remaining_percentage",
    },
    "baseline_position_count": {
        "type": "regression",
        "input_type": "position",
        "output_dim": 1,
        "target_key": "remaining_token_counts",
    },
    # Predict total completion length T from the prompt's last hidden state.
    # Trained on the prompt-end position only (target masked elsewhere); at
    # evaluation it is run at every position to test whether the prompt-trained
    # direction transfers to mid-completion residuals.
    "prompt_only_count": {
        "type": "regression",
        "input_type": "hidden_states",
        "output_dim": 1,
        "target_key": "total_completion_length_at_prompt_end",
        "train_position": "prompt_end",
    },
}
