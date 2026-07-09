"""Render local SINS contexts into paper-format prompt records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prompts.prompt_template import get_interpretation_matching_prompt_templates


ROOT = Path(__file__).resolve().parents[1]
FILLINGS_PATH = ROOT / "prompts" / "prompt_fillings.json"
TEXT_ACCESS_IDS = {
    "startUntilCurLine",
    "startUntilCurTranx",
    "curTranxUntilCurLine",
    "curTranxUntilCurTranx",
}
MAP_ACCESS_IDS = {"no_maps", "both_maps", "giver_only", "follower_only"}


def load_fillings(path: Path = FILLINGS_PATH) -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = {row["id"]: row["text"] for row in data["text_access_options"]}
    maps = {row["id"]: row["text"] for row in data["map_access_options"]}
    if set(text) != TEXT_ACCESS_IDS or set(maps) != MAP_ACCESS_IDS:
        raise ValueError("prompt fillings do not match the published condition IDs")
    return text, maps


def render_record(
    instance: dict[str, Any],
    context: str,
    text_access: str,
    map_access: str,
    maps: list[str] | None = None,
) -> dict[str, Any]:
    """Build one JSONL-ready record without copying caller-owned map files."""
    text_fillings, map_fillings = load_fillings()
    if text_access not in text_fillings or map_access not in map_fillings:
        raise ValueError("unknown text_access or map_access")
    if not isinstance(instance.get("ref_id"), str) or not instance["ref_id"]:
        raise ValueError("instance is missing ref_id")
    if not isinstance(instance.get("expression"), str) or not instance["expression"]:
        raise ValueError(f"{instance['ref_id']}: instance is missing expression")

    system_template, user_template = get_interpretation_matching_prompt_templates()
    return {
        "ref_id": instance["ref_id"],
        "system_prompt": system_template.substitute(
            text_access=text_fillings[text_access],
            map_access=map_fillings[map_access],
        ),
        "user_prompt": user_template.substitute(
            target_ref=instance["expression"],
            context=context,
        ),
        "maps": list(maps or []),
    }


def resolve_map_paths(map_id: str, map_access: str, map_dir: Path | None) -> list[str]:
    if map_access == "no_maps":
        return []
    if map_dir is None:
        raise ValueError("--map-dir is required when --map-access is not no_maps")

    sides = {
        "both_maps": ("g", "f"),
        "giver_only": ("g",),
        "follower_only": ("f",),
    }[map_access]
    number = str(map_id).lstrip("m")
    paths = [map_dir / f"map{number}{side}.png" for side in sides]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"local MapTask map files not found: {missing}")
    return [str(path) for path in paths]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render local SINS contexts into prompt JSONL.")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--text-access", choices=sorted(TEXT_ACCESS_IDS), required=True)
    parser.add_argument("--map-access", choices=sorted(MAP_ACCESS_IDS), required=True)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.instances.is_file() or not args.contexts.is_file():
        raise SystemExit("--instances and --contexts must point to existing local files")

    contexts = json.loads(args.contexts.read_text(encoding="utf-8"))
    instances = pd.read_parquet(args.instances).to_dict("records")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_file:
        for instance in instances:
            ref_id = instance["ref_id"]
            if ref_id not in contexts or args.text_access not in contexts[ref_id]:
                raise KeyError(f"missing {args.text_access} context for {ref_id}")
            maps = resolve_map_paths(instance["map_id"], args.map_access, args.map_dir)
            record = render_record(
                instance,
                contexts[ref_id][args.text_access],
                args.text_access,
                args.map_access,
                maps,
            )
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(instances)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
