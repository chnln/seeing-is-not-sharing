from scripts.vllm_yes_no_template import build_arg_parser, parse_yes_no


def test_paper_style_defaults_and_constrained_labels():
    args = build_arg_parser().parse_args(
        ["-i", "in.jsonl", "-o", "out.jsonl", "--model", "model-id"],
    )

    assert (args.temperature, args.seed, args.max_tokens) == (0.0, 42, 16)
    assert parse_yes_no("Yes") == "Yes"
    assert parse_yes_no(" no ") == "No"
