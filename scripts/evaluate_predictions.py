"""Evaluate SINS predictions keyed by ref_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_metrics(instances: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    required_instances = {"ref_id", "gold_label"}
    required_predictions = {"ref_id", "judge"}
    if not required_instances <= set(instances.columns):
        raise ValueError("instances must contain ref_id and gold_label")
    if not required_predictions <= set(predictions.columns):
        raise ValueError("predictions must contain ref_id and judge")
    if instances["ref_id"].duplicated().any() or predictions["ref_id"].duplicated().any():
        raise ValueError("instances and predictions must each have unique ref_id values")
    if set(instances["ref_id"]) != set(predictions["ref_id"]):
        raise ValueError("predictions must cover every SINS ref_id exactly once")

    merged = instances[["ref_id", "gold_label"]].merge(
        predictions[["ref_id", "judge"]],
        on="ref_id",
        validate="one_to_one",
    )
    gold = merged["gold_label"].map({"yes": 1, "no": 0})
    pred = merged["judge"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if gold.isna().any() or pred.isna().any():
        raise ValueError("gold labels and predictions must be Yes/No values")

    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    precision_yes, recall_yes = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
    precision_no, recall_no = safe_divide(tn, tn + fn), safe_divide(tn, tn + fp)
    f1_yes, f1_no = f1(precision_yes, recall_yes), f1(precision_no, recall_no)
    total = tp + fp + fn + tn
    return {
        "n": total,
        "accuracy": safe_divide(tp + tn, total),
        "yes_rate": safe_divide(tp + fp, total),
        "precision_yes": precision_yes,
        "recall_yes": recall_yes,
        "f1_yes": f1_yes,
        "precision_no": precision_no,
        "recall_no": recall_no,
        "f1_no": f1_no,
        "f1_macro": (f1_yes + f1_no) / 2,
    }


def read_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as predictions_file:
        for line_number, line in enumerate(predictions_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute basic SINS binary metrics.")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.instances.is_file() or not args.predictions.is_file():
        raise SystemExit("--instances and --predictions must point to existing files")
    metrics = compute_metrics(pd.read_parquet(args.instances), read_predictions(args.predictions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
