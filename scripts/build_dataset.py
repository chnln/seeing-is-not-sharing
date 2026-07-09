"""Build the compact SINS dataset from a local GMMT checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "ref_id",
    "dialogue_id",
    "map_id",
    "utt_id",
    "transaction_id",
    "context_transaction_ids",
    "end_utt_id_of_context",
    "timed_unit_ids",
    "expression",
    "status",
    "gold_label",
]
INPUT_COLUMNS = [
    "ref_id_unif",
    "dialogue_id",
    "map_id",
    "utt_id",
    "transaction_id",
    "context_transaction_ids",
    "end_utt_id_of_context",
    "timed_unit_ids",
    "expression",
    "status",
]
GOLD_LABELS = {"aligned": "yes", "pending": "no", "misunderstood": "no"}


def build_instances(re_level: pd.DataFrame) -> pd.DataFrame:
    """Project public SINS fields from GMMT's RE-level annotation table."""
    missing = [column for column in INPUT_COLUMNS if column not in re_level.columns]
    if missing:
        raise ValueError(f"GMMT re_level missing columns: {missing}")

    instances = re_level[INPUT_COLUMNS].copy().rename(columns={"ref_id_unif": "ref_id"})
    unknown = sorted(set(instances["status"].dropna()) - set(GOLD_LABELS))
    if unknown:
        raise ValueError(f"Unexpected GMMT status values: {unknown}")
    if instances["status"].isna().any():
        raise ValueError("GMMT status contains null values")

    instances["gold_label"] = instances["status"].map(GOLD_LABELS)
    return instances[OUTPUT_COLUMNS]


def validate_instances(instances: pd.DataFrame, require_full: bool = False) -> None:
    """Validate the public schema and, optionally, the released corpus totals."""
    if list(instances.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Public columns must be exactly {OUTPUT_COLUMNS}")
    if instances["ref_id"].isna().any() or instances["ref_id"].duplicated().any():
        raise ValueError("ref_id must be present and unique")

    expected_gold = instances["status"].map(GOLD_LABELS)
    if expected_gold.isna().any() or not expected_gold.equals(instances["gold_label"]):
        raise ValueError("gold_label does not match status")

    if require_full:
        counts = instances["gold_label"].value_counts().to_dict()
        if len(instances) != 13077 or counts != {"yes": 9435, "no": 3642}:
            raise ValueError(f"Unexpected SINS totals: n={len(instances)}, counts={counts}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the compact SINS HF dataset from a local GMMT checkout.",
    )
    parser.add_argument("--gmmt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source = args.gmmt_dir / "data" / "re_level" / "train.parquet"
    if not source.is_file():
        raise SystemExit(f"GMMT re_level parquet not found: {source}")

    instances = build_instances(pd.read_parquet(source))
    validate_instances(instances, require_full=True)

    target = args.out_dir / "data" / "train-00000-of-00001.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    instances.to_parquet(target, index=False)
    print(f"wrote {len(instances)} SINS rows -> {target}")


if __name__ == "__main__":
    main()
