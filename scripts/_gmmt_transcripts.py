#!/usr/bin/env python3
"""Reconstruct the RE-marked MapTask transcripts (`*_all_refs.txt`) locally.

The public release does NOT redistribute MapTask transcript text (the HCRC Map
Task Corpus forbids onward distribution of corpus materials). Instead, this
self-contained script rebuilds the marked transcripts from:

  1. YOUR own copy of the raw MapTask timed-units XML (download separately, see
     README), and
  2. the reference-expression files shipped with this dataset
     (`reference_expressions/{dialogue}.reference_expressions.json`), which carry
     only our annotations: unified timed-unit IDs, ref IDs and landmark IDs.

It reproduces the exact upstream pipeline (fill missing `utt` → merge giver +
follower timed-units → sort by start time → assign `{dialogue}.tu.{i}` unified
IDs → group into utterance lines → insert `<<expression id:REF lm:LM>>` markers),
so the output matches the original `*_all_refs.txt` byte-for-byte.

Where to put the MapTask download:
  After downloading the HCRC Map Task Corpus (v2.1) and unpacking it, point
  `--maptask-tu-dir` at the directory that directly contains the per-side files
  named `<dialogue>.g.timed-units.xml` and `<dialogue>.f.timed-units.xml`
  (in the v2.1 distribution: `maptaskv2-1/Data/timed-units/`).
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json

NITE = "http://nite.sourceforge.net/"
UTT_DIGITS = re.compile(r"\d+")


# --------------------------------------------------------------------------- #
# 1. Parse one raw per-side timed-units XML into ordered units.
# --------------------------------------------------------------------------- #
def parse_side(xml_path: Path, speaker: str) -> List[dict]:
    """Return `<tu>` units in document order: {tu_id, start, end, utt, text, speaker}.

    `utt` is an int when present on the element, else None (filled later).
    Non-`tu` elements (sil/noi) are ignored, matching the upstream transform.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    units: List[dict] = []
    for tu in root.findall(".//tu", {"nite": NITE}):
        utt_raw = tu.get("utt")
        utt = None
        if utt_raw is not None:
            m = UTT_DIGITS.search(utt_raw)
            if m:
                utt = int(m.group(0))
        units.append({
            "tu_id": tu.get("id", ""),
            "start_time": float(tu.get("start", 0) or 0),
            "end_time": float(tu.get("end", 0) or 0),
            "utt": utt,
            "text": tu.text or "",
            "speaker": speaker,
        })
    return units


# --------------------------------------------------------------------------- #
# 2. Fill missing `utt` (port of scripts/fix_missing_utt.py, in-memory).
# --------------------------------------------------------------------------- #
def fill_missing_utt(units: List[dict]) -> None:
    """Fill `utt` gaps in place using nearest labeled neighbors + time tie-break.

    Only fills missing values; never renumbers existing ones (deterministic).
    """
    n = len(units)

    def prev_labeled(i: int) -> Optional[int]:
        j = i - 1
        while j >= 0:
            if units[j]["utt"] is not None:
                return j
            j -= 1
        return None

    def next_labeled(i: int) -> Optional[int]:
        j = i + 1
        while j < n:
            if units[j]["utt"] is not None:
                return j
            j += 1
        return None

    for i, u in enumerate(units):
        if u["utt"] is not None:
            continue
        p, q = prev_labeled(i), next_labeled(i)
        if p is not None and q is not None:
            up, uq = units[p]["utt"], units[q]["utt"]
            if up == uq:
                u["utt"] = up
            else:
                ts, ps, qs = u["start_time"], units[p]["start_time"], units[q]["start_time"]
                u["utt"] = up if abs(ts - ps) <= abs(qs - ts) else uq
        elif p is not None:
            u["utt"] = units[p]["utt"]
        elif q is not None:
            u["utt"] = units[q]["utt"]
        # else: all-unknown -> leave as None (becomes -1 downstream)


# --------------------------------------------------------------------------- #
# 3. Merge giver + follower, sort, assign unified tu IDs (port of notebook 1).
# --------------------------------------------------------------------------- #
def build_unified_units(dialogue_id: str, giver_units: List[dict], follower_units: List[dict]) -> List[dict]:
    all_units = list(giver_units) + list(follower_units)   # giver first (stable tie-break)
    all_units.sort(key=lambda x: x.get("start_time", 0))    # stable sort
    for i, unit in enumerate(all_units):
        unit["tu_id_unif"] = f"{dialogue_id}.tu.{i}"
        unit["utt_id"] = unit["utt"] if unit["utt"] is not None else -1
    return all_units


# --------------------------------------------------------------------------- #
# 4. Render marked transcript (port of notebook 2 all_refs generator).
# --------------------------------------------------------------------------- #
def render_transcript(dialogue_id: str, map_id: str, units: List[dict], refs: List[dict]) -> str:
    # tu_id_unif -> list of {is_start, is_end, ref_id, lm_id, ref}
    tu_ref_info: Dict[str, List[dict]] = defaultdict(list)
    for ref in refs:
        tu_ids = ref.get("timed_unit_ids", [])
        for i, tu_id in enumerate(tu_ids):
            tu_ref_info[tu_id].append({
                "is_start": i == 0,
                "is_end": i == len(tu_ids) - 1,
                "ref_id": ref["ref_id_unif"],
                "lm_id": ref["lm_id_orig"],
                "start_time": ref.get("start_time", 0),
            })

    # group units into utterance lines
    groups: List[dict] = []
    current: Optional[dict] = None
    for u in units:
        speaker = u.get("speaker", "")
        utt_id = int(u.get("utt_id", -1))
        text = " ".join((u.get("text", "") or "").strip().split())
        tu_unif = u.get("tu_id_unif", "")
        if current is not None and (
            (utt_id != -1 and current["utt_id"] == utt_id and current["speaker"] == speaker)
            or (utt_id == -1 and current["utt_id"] == -1 and current["speaker"] == speaker)
        ):
            current["parts"].append((text, tu_unif))
        else:
            if current is not None:
                groups.append(current)
            current = {"speaker": speaker, "utt_id": utt_id, "parts": [(text, tu_unif)]}
    if current is not None:
        groups.append(current)

    lines = [f"=== Dialogue: {dialogue_id} ===", f"Map ID: m{map_id}", ""]
    for i, group in enumerate(groups, 1):
        segments: List[str] = []
        for text, tu_id in group["parts"]:
            if not text:
                continue
            if tu_id in tu_ref_info:
                infos = sorted(tu_ref_info[tu_id], key=lambda x: x["start_time"])
                segment = text
                for info in [x for x in infos if x["is_start"]]:
                    segment = f"<<{segment}"
                for info in [x for x in infos if x["is_end"]]:
                    segment = f"{segment} id:{info['ref_id']} lm:{info['lm_id']}>>"
                segments.append(segment)
            else:
                segments.append(text)
        highlighted = " ".join(segments)
        label = f"[{group['speaker']:<8} ln:{i:<3} utt:{group['utt_id']:<3}]"
        lines.append(f"{label} {highlighted}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_map_lookup(mapping_path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if not toks:
            continue
        map_id, dialogues = toks[0], toks[1:]
        for d in dialogues:
            lookup[d] = map_id
    return lookup


def default_map_mapping_path(repo_root: Path) -> Path:
    # Vendored upstream helper; unused by the SINS adapter, which passes map_id directly.
    # The HF release flattens metadata under assets/; the GitHub release keeps
    # assets under annotations/. Prefer the HF layout, then fall back to GitHub.
    hf_layout = repo_root / "assets/map_trans_mapping.txt"
    if hf_layout.exists():
        return hf_layout
    return repo_root / "annotations/assets/map_trans_mapping.txt"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maptask-tu-dir", required=True,
                   help="dir with raw <dialogue>.{g,f}.timed-units.xml (your MapTask download)")
    p.add_argument("--re-dir", default=str(here / "reference_expressions"),
                   help="reference-expression JSON dir shipped with this dataset")
    p.add_argument("--map-mapping", default=str(default_map_mapping_path(here)),
                   help="map_trans_mapping.txt (shipped asset) for the Map ID header")
    p.add_argument("--out-dir", required=True, help="output dir for reconstructed *_all_refs.txt")
    p.add_argument("--dialogue-ids", nargs="*", help="subset of dialogue IDs (default: all found)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tu_dir = Path(args.maptask_tu_dir)
    re_dir = Path(args.re_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_lookup = load_map_lookup(Path(args.map_mapping))

    if args.dialogue_ids:
        dialogue_ids = args.dialogue_ids
    else:
        dialogue_ids = sorted(
            p.name[: -len(".g.timed-units.xml")]
            for p in tu_dir.glob("*.g.timed-units.xml")
        )

    written = 0
    for dialogue_id in dialogue_ids:
        g_path = tu_dir / f"{dialogue_id}.g.timed-units.xml"
        f_path = tu_dir / f"{dialogue_id}.f.timed-units.xml"
        re_path = re_dir / f"{dialogue_id}.reference_expressions.json"
        if not (g_path.exists() and f_path.exists() and re_path.exists()):
            if args.verbose:
                print(f"[skip] missing inputs for {dialogue_id}")
            continue

        giver = parse_side(g_path, "giver")
        follower = parse_side(f_path, "follower")
        fill_missing_utt(giver)
        fill_missing_utt(follower)
        units = build_unified_units(dialogue_id, giver, follower)

        refs = json.loads(re_path.read_text(encoding="utf-8")).get("landmark_reference_expressions", [])
        map_id = map_lookup.get(dialogue_id, "unknown")
        text = render_transcript(dialogue_id, map_id, units, refs)

        (out_dir / f"{dialogue_id}_all_refs.txt").write_text(text, encoding="utf-8")
        written += 1
        if args.verbose:
            print(f"[ok] {dialogue_id} ({len(refs)} refs)")

    print(f"reconstructed {written} transcript(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
