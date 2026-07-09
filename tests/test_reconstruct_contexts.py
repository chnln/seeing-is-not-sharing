import numpy as np

from scripts.reconstruct_contexts import (
    build_utt_to_transaction,
    clean_ref_markers,
    extract_transaction_until_ln,
)


def test_clean_markers_keeps_target_and_strips_other_reference_markers():
    text = "<<the lake id:d1.ref.0 lm:m1_lake>> near <<the road id:d1.ref.1 lm:m1_road>>"

    assert clean_ref_markers(text, "d1.ref.0") == "<<the lake>> near the road"


def test_clean_markers_preserves_nested_overlap_as_in_the_original_builder():
    text = "<<<<they id:d1.ref.45 lm:m1_a>> id:d1.ref.46 lm:m1_b>>"

    assert clean_ref_markers(text, "d1.ref.45") == "<<<<they>> id:d1.ref.46 lm:m1_b>>"


def test_transaction_window_uses_transaction_utterance_membership():
    lines = [
        "=== Dialogue: q1 ===",
        "Map ID: m0",
        "",
        "[giver    ln:11  utt:11 ] old transaction continuation",
        "[follower ln:14  utt:12 ] current transaction",
        "[giver    ln:15  utt:13 ] <<the meadow id:q1.ref.4 lm:m0_meadow>>",
    ]

    context = extract_transaction_until_ln(lines, [11, 12, 13], target_ln=15)

    assert "old transaction continuation" in context
    assert "current transaction" in context
    assert "the meadow" in context


def test_transaction_index_accepts_parquet_style_numpy_utterance_arrays():
    transactions = [{"transaction_id": "t1", "utt_ids": np.array([1, 2])}]

    assert build_utt_to_transaction(transactions) == {1: 0, 2: 0}
