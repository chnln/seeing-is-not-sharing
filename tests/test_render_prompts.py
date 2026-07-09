from scripts.render_prompts import render_record


INSTANCE = {"ref_id": "q1ec1.ref.0", "expression": "a caravan park"}
CONTEXT = "=== Dialogue: q1ec1 ===\nMap ID: m12\n\n\n[giver    ln:1   utt:1  ] <<a caravan park>>"


def test_text_only_record_uses_paper_template():
    record = render_record(INSTANCE, CONTEXT, "startUntilCurLine", "no_maps")

    assert record["ref_id"] == "q1ec1.ref.0"
    assert "Answer with exactly one word: Yes or No." in record["system_prompt"]
    assert "No map images are provided." in record["system_prompt"]
    assert CONTEXT in record["user_prompt"]
    assert record["maps"] == []


def test_both_maps_keeps_user_owned_paths_outside_prompt_text():
    paths = ["/tmp/map12g.png", "/tmp/map12f.png"]
    record = render_record(INSTANCE, CONTEXT, "startUntilCurTranx", "both_maps", paths)

    assert "both the Giver's and the Follower's map images" in record["system_prompt"]
    assert record["maps"] == paths
    assert "/tmp/map12g.png" not in record["user_prompt"]
