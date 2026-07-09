from pathlib import Path


def test_release_has_only_small_public_interfaces():
    root = Path(__file__).resolve().parents[1]
    for path in (
        "scripts/build_dataset.py",
        "scripts/reconstruct_contexts.py",
        "scripts/render_prompts.py",
        "scripts/vllm_yes_no_template.py",
        "scripts/evaluate_predictions.py",
        "prompts/prompt_template.py",
        "prompts/prompt_fillings.json",
    ):
        assert (root / path).is_file(), path
    for removed in ("analysis", "experiments", "src", "config.sh"):
        assert not (root / removed).exists(), removed
