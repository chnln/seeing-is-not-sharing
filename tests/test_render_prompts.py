from scripts.render_prompts import render_record
from pathlib import Path


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


def test_appendix_textual_map_examples_are_exact_fixtures():
    fixtures = Path(__file__).resolve().parents[1] / "prompts" / "fixtures"
    landmark_names = fixtures.joinpath("q1ec1_m12_landmark_names.txt").read_text(encoding="utf-8")
    discrepancy_detail = fixtures.joinpath("q1ec1_m12_discrepancy_detail.txt").read_text(encoding="utf-8")

    expected_landmark_names = (
        "Map landmark information:\n"
        "- Giver's map landmarks: start, caravan park, old mill, abandoned cottage, fenced meadow, fenced meadow, west lake, trig point, monument, nuclear test site, east lake, farmed land, finish\n"
        "- Follower's map landmarks: start, caravan park, picket fence, mill wheel, forest, abandoned cottage, fenced meadow, west lake, monument, golf course, east lake, farmed land\n"
    )
    expected_discrepancy_detail = expected_landmark_names + (
        "Discrepancies between maps:\n"
        "- Landmarks on Giver's map ONLY (not on Follower's): finish, nuclear test site, old mill, trig point\n"
        "- Landmarks on Follower's map ONLY (not on Giver's): forest, golf course, mill wheel, picket fence\n"
        "- Landmarks appearing multiple times: fenced meadow appears 2 times on Giver's map\n"
        "- Shared landmarks (on both maps): abandoned cottage, caravan park, east lake, farmed land, fenced meadow, monument, start, west lake\n"
    )

    assert landmark_names == expected_landmark_names
    assert discrepancy_detail == expected_discrepancy_detail
