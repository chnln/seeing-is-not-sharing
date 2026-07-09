from pathlib import Path


def test_readme_and_dataset_card_state_non_redistribution_boundary():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    card = (root / "dataset_card.md").read_text(encoding="utf-8").lower()

    for phrase in (
        "does not contain dialogue context",
        "does not contain maptask maps",
        "image-derived text",
    ):
        assert phrase in readme
        assert phrase in card
    assert "grounded-misunderstandings-in-maptask" in readme
    assert "reproduces the paper's figures" not in readme


def test_dataset_card_has_hf_license_and_notice_names_gmmt():
    root = Path(__file__).resolve().parents[1]

    assert "license: cc-by-4.0" in (root / "dataset_card.md").read_text(encoding="utf-8").lower()
    assert "grounded-misunderstandings-in-maptask" in (root / "NOTICE").read_text(encoding="utf-8")
