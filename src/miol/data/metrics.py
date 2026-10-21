"""
Metrics computation for token count prediction.
"""


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for token count prediction.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids

    Returns:
        Dictionary of metrics (mae, mse, max_error, etc.)
    """
    predictions = eval_pred.predictions  # token_count_predictions
    labels = eval_pred.label_ids  # remaining_token_counts

    # Flatten arrays and filter out masked positions (-100)
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()

    # Only compute metrics on non-masked positions
    mask = labels_flat != -100
    valid_predictions = predictions_flat[mask]
    valid_labels = labels_flat[mask]

    # Compute metrics (handle case where all sequences were truncated)
    if len(valid_predictions) > 0:
        errors = abs(valid_predictions - valid_labels)
        mae = float(errors.mean())
        mse = float((errors ** 2).mean())
        max_error = float(errors.max())
    else:
        # All sequences truncated (no valid targets) - return NaN to exclude from averages
        mae = float('nan')
        mse = float('nan')
        max_error = float('nan')

    return {
        "mae": mae,
        "mse": mse,
        "max_error": max_error,
    }
