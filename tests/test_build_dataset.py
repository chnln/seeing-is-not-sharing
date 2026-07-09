import pandas as pd
import pytest

from scripts.build_dataset import OUTPUT_COLUMNS, build_instances, validate_instances


def _re_level():
    return pd.DataFrame([
        {
            "ref_id_unif": "q1.ref.0",
            "dialogue_id": "q1",
            "map_id": "m0",
            "utt_id": 1,
            "transaction_id": "q1.transaction.1",
            "context_transaction_ids": ["q1.transaction.1"],
            "end_utt_id_of_context": 2,
            "timed_unit_ids": ["q1.tu.0"],
            "expression": "the lake",
            "status": "aligned",
        },
        {
            "ref_id_unif": "q1.ref.1",
            "dialogue_id": "q1",
            "map_id": "m0",
            "utt_id": 2,
            "transaction_id": "q1.transaction.1",
            "context_transaction_ids": ["q1.transaction.1"],
            "end_utt_id_of_context": 2,
            "timed_unit_ids": ["q1.tu.1"],
            "expression": "the house",
            "status": "pending",
        },
        {
            "ref_id_unif": "q1.ref.2",
            "dialogue_id": "q1",
            "map_id": "m0",
            "utt_id": 2,
            "transaction_id": "q1.transaction.1",
            "context_transaction_ids": ["q1.transaction.1"],
            "end_utt_id_of_context": 2,
            "timed_unit_ids": ["q1.tu.2"],
            "expression": "the road",
            "status": "misunderstood",
        },
    ])


def test_build_keeps_only_public_schema_and_derives_binary_gold():
    out = build_instances(_re_level())

    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["ref_id"].tolist() == ["q1.ref.0", "q1.ref.1", "q1.ref.2"]
    assert out["gold_label"].tolist() == ["yes", "no", "no"]
    assert "reason" not in out.columns
    assert "giver_interpretation" not in out.columns


def test_invalid_status_is_rejected():
    source = _re_level()
    source.loc[0, "status"] = "unknown"

    with pytest.raises(ValueError, match="status"):
        build_instances(source)


def test_full_validation_checks_published_totals():
    source = pd.concat(
        [_re_level().iloc[[0]]] * 9435
        + [_re_level().iloc[[1]]] * 3403
        + [_re_level().iloc[[2]]] * 239,
        ignore_index=True,
    )
    source["ref_id_unif"] = [f"q.ref.{index}" for index in range(len(source))]

    validate_instances(build_instances(source), require_full=True)
