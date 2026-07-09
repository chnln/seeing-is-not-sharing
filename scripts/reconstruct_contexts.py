"""Reconstruct local SINS contexts from GMMT and a user's MapTask copy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import _gmmt_transcripts as gmmt


TEXT_ACCESS = (
    "startUntilCurLine",
    "startUntilCurTranx",
    "curTranxUntilCurLine",
    "curTranxUntilCurTranx",
)
MANUAL_REF_TO_TRANSACTION = {"q4ec4.ref.61": "q4ec4.transaction.11"}


def clean_ref_markers(text: str, target_ref_id: str) -> str:
    """Keep the target RE bracketed and remove all other RE markers."""
    pattern = r"<<([\s\S]+?)\s+id:([^\s]+)\s+lm:([^\s]+)>>"

    def replace(match: re.Match[str]) -> str:
        ref_text = match.group(1).strip()
        return f"<<{ref_text}>>" if match.group(2).strip() == target_ref_id else ref_text

    return re.sub(pattern, replace, text)


def _header(lines: list[str]) -> list[str]:
    return [line for index, line in enumerate(lines[:3]) if index < 3 and (line.startswith("===") or line.startswith("Map ID:") or not line.strip())]


def _line_number(line: str) -> int | None:
    match = re.search(r"ln:(\d+)", line)
    return int(match.group(1)) if match else None


def _utterance_id(line: str) -> int | None:
    match = re.search(r"utt:(\d+)", line)
    return int(match.group(1)) if match else None


def find_target_line(lines: list[str], ref_id: str) -> int | None:
    for line in lines:
        if f"id:{ref_id}" in line:
            return _line_number(line)
    return None


def find_target_utterance(lines: list[str], ref_id: str) -> int | None:
    for line in lines:
        if f"id:{ref_id}" in line:
            return _utterance_id(line)
    return None


def extract_lines_until_ln(lines: list[str], target_ln: int) -> str:
    content: list[str] = []
    for line in lines[3:]:
        line_number = _line_number(line)
        if line_number is None:
            continue
        if line_number <= target_ln:
            content.append(line)
        else:
            break
    return "\n".join(_header(lines) + [""] + content).strip()


def extract_transaction_lines(lines: list[str], utt_ids: list[int]) -> list[str]:
    allowed = {int(utt_id) for utt_id in utt_ids}
    return [line for line in lines if _utterance_id(line) in allowed]


def extract_transaction_until_ln(lines: list[str], utt_ids: list[int], target_ln: int) -> str:
    content = [
        line
        for line in extract_transaction_lines(lines, utt_ids)
        if (_line_number(line) or 0) <= target_ln
    ]
    return "\n".join(_header(lines) + [""] + content).strip()


def extract_transaction_context(lines: list[str], utt_ids: list[int]) -> str:
    return "\n".join(_header(lines) + [""] + extract_transaction_lines(lines, utt_ids)).strip()


def transaction_utt_ids(transaction: dict[str, Any]) -> list[int]:
    value = transaction.get("utt_ids")
    return [] if value is None else [int(utt_id) for utt_id in value]


def build_utt_to_transaction(transactions: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, transaction in enumerate(transactions):
        for utt_id in transaction_utt_ids(transaction):
            mapping[utt_id] = index
    return mapping


def find_transaction_by_id(transactions: list[dict[str, Any]], transaction_id: str) -> int | None:
    for index, transaction in enumerate(transactions):
        if transaction.get("transaction_id") == transaction_id:
            return index
    return None


def find_transaction_by_line(
    lines: list[str],
    transactions: list[dict[str, Any]],
    target_ln: int,
) -> int | None:
    best: tuple[int, int] | None = None
    for index, transaction in enumerate(transactions):
        transaction_lines = extract_transaction_lines(lines, transaction_utt_ids(transaction))
        line_numbers = [line_number for line in transaction_lines if (line_number := _line_number(line)) is not None]
        if not line_numbers:
            continue
        low, high = min(line_numbers), max(line_numbers)
        if low <= target_ln <= high and (best is None or high - low < best[1]):
            best = (index, high - low)
    return None if best is None else best[0]


def context_windows(
    instance: dict[str, Any],
    all_display_lines: list[str],
    transactions: list[dict[str, Any]],
    utt_to_transaction: dict[int, int],
) -> dict[str, str]:
    ref_id = instance["ref_id"]
    target_ln = find_target_line(all_display_lines, ref_id)
    if target_ln is None:
        raise ValueError(f"target marker not found in reconstructed transcript: {ref_id}")

    target_utt = find_target_utterance(all_display_lines, ref_id)
    transaction_index = None
    override = MANUAL_REF_TO_TRANSACTION.get(ref_id)
    if override:
        transaction_index = find_transaction_by_id(transactions, override)
    annotated_utt = int(instance["utt_id"])
    if transaction_index is None:
        transaction_index = utt_to_transaction.get(annotated_utt)
    if transaction_index is None and target_utt is not None:
        transaction_index = utt_to_transaction.get(target_utt)
    if transaction_index is None:
        transaction_index = find_transaction_by_line(all_display_lines, transactions, target_ln)
    if transaction_index is None:
        raise ValueError(f"transaction not found for {ref_id}")

    transaction_utts = transaction_utt_ids(transactions[transaction_index])
    candidate_utt = target_utt if target_utt is not None else annotated_utt
    if candidate_utt not in transaction_utts:
        transaction_utts = sorted(set(transaction_utts + [candidate_utt]))

    max_transaction_ln = max(
        (_line_number(line) or 0 for line in extract_transaction_lines(all_display_lines, transaction_utts)),
        default=target_ln,
    )
    raw_windows = {
        "startUntilCurLine": extract_lines_until_ln(all_display_lines, target_ln),
        "startUntilCurTranx": extract_lines_until_ln(all_display_lines, max_transaction_ln),
        "curTranxUntilCurLine": extract_transaction_until_ln(all_display_lines, transaction_utts, target_ln),
        "curTranxUntilCurTranx": extract_transaction_context(all_display_lines, transaction_utts),
    }
    return {name: clean_ref_markers(text, ref_id) for name, text in raw_windows.items()}


def _reference_expressions_path(gmmt_dir: Path, dialogue_id: str) -> Path:
    path = gmmt_dir / "reference_expressions" / f"{dialogue_id}.reference_expressions.json"
    if not path.is_file():
        raise FileNotFoundError(f"GMMT reference-expression file not found: {path}")
    return path


def full_transcript(
    dialogue_id: str,
    map_id: str,
    maptask_tu_dir: Path,
    gmmt_dir: Path,
) -> str:
    """Use GMMT's public algorithm to regenerate the all-RE display transcript."""
    giver_path = maptask_tu_dir / f"{dialogue_id}.g.timed-units.xml"
    follower_path = maptask_tu_dir / f"{dialogue_id}.f.timed-units.xml"
    for path in (giver_path, follower_path):
        if not path.is_file():
            raise FileNotFoundError(f"MapTask timed-units file not found: {path}")

    giver = gmmt.parse_side(giver_path, "giver")
    follower = gmmt.parse_side(follower_path, "follower")
    gmmt.fill_missing_utt(giver)
    gmmt.fill_missing_utt(follower)
    refs = json.loads(_reference_expressions_path(gmmt_dir, dialogue_id).read_text(encoding="utf-8"))
    return gmmt.render_transcript(
        dialogue_id,
        str(map_id).lstrip("m"),
        gmmt.build_unified_units(dialogue_id, giver, follower),
        refs["landmark_reference_expressions"],
    )


def reconstruct_all(
    instances: list[dict[str, Any]],
    skeletons: dict[str, dict[str, Any]],
    maptask_tu_dir: str | Path,
    gmmt_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Return all four local context windows for every SINS instance."""
    tu_dir = Path(maptask_tu_dir)
    companion_dir = Path(gmmt_dir)
    cache: dict[str, tuple[list[str], list[dict[str, Any]], dict[int, int]]] = {}
    contexts: dict[str, dict[str, str]] = {}

    for instance in instances:
        dialogue_id = instance["dialogue_id"]
        if dialogue_id not in cache:
            skeleton = skeletons[dialogue_id]
            transcript = full_transcript(dialogue_id, instance["map_id"], tu_dir, companion_dir)
            transactions = list(skeleton["transactions"])
            cache[dialogue_id] = (
                transcript.splitlines(),
                transactions,
                build_utt_to_transaction(transactions),
            )
        all_display_lines, transactions, utt_to_transaction = cache[dialogue_id]
        contexts[instance["ref_id"]] = context_windows(
            instance,
            all_display_lines,
            transactions,
            utt_to_transaction,
        )
    return contexts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct local SINS contexts from GMMT and MapTask v2.1.",
    )
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--gmmt-dir", type=Path, required=True)
    parser.add_argument("--maptask-tu-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.instances.is_file():
        raise SystemExit(f"SINS instances parquet not found: {args.instances}")
    if not args.maptask_tu_dir.is_dir():
        raise SystemExit(f"MapTask timed-units directory not found: {args.maptask_tu_dir}")

    dialogue_path = args.gmmt_dir / "data" / "dialogue_level" / "train.parquet"
    if not dialogue_path.is_file():
        raise SystemExit(f"GMMT dialogue-level parquet not found: {dialogue_path}")

    instances = pd.read_parquet(args.instances).to_dict("records")
    dialogues = pd.read_parquet(dialogue_path).to_dict("records")
    skeletons = {dialogue["dialogue_id"]: dialogue for dialogue in dialogues}
    contexts = reconstruct_all(instances, skeletons, args.maptask_tu_dir, args.gmmt_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contexts, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(contexts)} reconstructed instances -> {args.out}")


if __name__ == "__main__":
    main()
