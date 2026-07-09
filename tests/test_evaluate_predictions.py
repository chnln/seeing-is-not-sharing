import pandas as pd

from scripts.evaluate_predictions import compute_metrics


def test_binary_metrics_join_on_ref_id():
    instances = pd.DataFrame(
        {"ref_id": ["a", "b", "c", "d"], "gold_label": ["yes", "yes", "no", "no"]},
    )
    predictions = pd.DataFrame(
        {"ref_id": ["a", "b", "c", "d"], "judge": ["Yes", "No", "No", "Yes"]},
    )

    metrics = compute_metrics(instances, predictions)

    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["f1_macro"] == 0.5
