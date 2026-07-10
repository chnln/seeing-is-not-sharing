from pathlib import Path


SINS_GITHUB = "https://github.com/chnln/seeing-is-not-sharing"
SINS_HF = "https://huggingface.co/datasets/chnln/seeing-is-not-sharing"
SINS_PAPER = "https://arxiv.org/abs/2606.31719"
GMMT_GITHUB = "https://github.com/chnln/grounded-misunderstandings-in-maptask"
GMMT_HF = "https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask"
GMMT_PAPER = "https://arxiv.org/abs/2511.03718"


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


def test_public_documents_link_the_task_papers_and_related_releases():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    card = (root / "dataset_card.md").read_text(encoding="utf-8")
    task_definition = "whether the giver and follower interpret the referring expression as the same landmark"
    normalise = lambda text: " ".join(text.lower().split())

    assert task_definition in normalise(readme)
    assert task_definition in normalise(card)
    assert "SIGDIAL 2026" in readme
    assert "SIGDIAL 2026" in card
    for link in (SINS_PAPER, GMMT_GITHUB, GMMT_HF, GMMT_PAPER):
        assert link in readme
    for link in (SINS_GITHUB, SINS_PAPER, GMMT_GITHUB, GMMT_HF, GMMT_PAPER):
        assert link in card


def test_readme_uses_public_cli_arguments_without_git_implementation_details():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8").lower()

    assert "--out-dir release/hf" in readme
    assert "--text-access startuntilcurline" in readme
    assert "--map-access no_maps" in readme
    assert "--maptask-tu-dir /path/to/maptaskv2-1/data/timed-units" in readme
    assert "timed-units_utt_filled" not in readme
    assert "git clone https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask" in readme
    assert "pip install -e \".[dev]\"" in readme
    assert "--condition no_maps" not in readme
    assert "ignored by git" not in readme


def test_dataset_card_shows_the_standard_hf_loading_path():
    card = (Path(__file__).resolve().parents[1] / "dataset_card.md").read_text(encoding="utf-8")

    assert "from datasets import load_dataset" in card
    assert 'load_dataset("chnln/seeing-is-not-sharing", split="train")' in card


def test_dataset_card_has_hf_license_and_notice_names_gmmt():
    root = Path(__file__).resolve().parents[1]

    assert "license: cc-by-4.0" in (root / "dataset_card.md").read_text(encoding="utf-8").lower()
    assert "grounded-misunderstandings-in-maptask" in (root / "NOTICE").read_text(encoding="utf-8")
