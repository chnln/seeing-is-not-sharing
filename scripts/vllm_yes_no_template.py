"""Example constrained Yes/No inference for SINS prompt JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run constrained Yes/No inference over local SINS prompt JSONL.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser


def parse_yes_no(text: str) -> str:
    match = YES_NO_PATTERN.search((text or "").strip())
    if not match:
        raise ValueError(f"model output is not Yes/No: {text!r}")
    return "Yes" if match.group(1).lower() == "yes" else "No"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def prompt_record(value: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    ref_id = value.get("ref_id")
    system_prompt = value.get("system_prompt")
    user_prompt = value.get("user_prompt")
    maps = value.get("maps", [])
    if not isinstance(ref_id, str) or not ref_id:
        raise ValueError("prompt record is missing ref_id")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise ValueError(f"{ref_id}: prompt record is missing text")
    if not isinstance(maps, list) or not all(isinstance(path, str) for path in maps):
        raise ValueError(f"{ref_id}: maps must be a list of paths")
    return ref_id, system_prompt, user_prompt, maps


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_tokens <= 0:
        raise SystemExit("--batch-size and --max-tokens must be positive")
    if not args.input.is_file():
        raise SystemExit(f"input JSONL not found: {args.input}")

    try:
        from PIL import Image
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise SystemExit("Install the optional dependency group first: uv sync --extra vllm") from error

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.limit_mm_per_prompt > 0:
        llm_kwargs["limit_mm_per_prompt"] = {"image": args.limit_mm_per_prompt}
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(choice=["Yes", "No"]),
    )

    records = [prompt_record(value) for value in iter_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            conversations = []
            for ref_id, system_prompt, user_prompt, maps in batch:
                if maps:
                    if args.limit_mm_per_prompt < len(maps):
                        raise ValueError(f"{ref_id}: --limit-mm-per-prompt is smaller than maps")
                    content: str | list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
                    for map_path in maps:
                        path = Path(map_path)
                        if not path.is_file():
                            raise FileNotFoundError(f"{ref_id}: map path not found: {path}")
                        with Image.open(path) as image:
                            content.append({"type": "image_pil", "image_pil": image.convert("RGB").copy()})
                else:
                    content = user_prompt
                conversations.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ])

            kwargs: dict[str, Any] = {"sampling_params": sampling, "use_tqdm": False}
            if args.disable_thinking:
                kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            outputs = llm.chat(conversations, **kwargs)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs than inputs")
            for (ref_id, _, _, _), output in zip(batch, outputs, strict=True):
                text = output.outputs[0].text if output.outputs else ""
                output_file.write(json.dumps({"ref_id": ref_id, "judge": parse_yes_no(text)}) + "\n")


if __name__ == "__main__":
    main()
