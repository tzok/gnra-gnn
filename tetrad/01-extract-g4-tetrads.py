#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 01 — extract G-tetrads from ElTetrado JSON annotations.

Reads every ``<pdb>-assembly<N>.json`` produced by ElTetrado from
``ONQUADRO_JSON_DIR`` (default: the ``json/`` directory of the local
OnQuadro mirror), keeps only tetrads whose four nucleotides are all guanines
(``shortName == "G"`` — this covers both RNA ``G`` and DNA ``DG``), drops PDB
entries listed in ``blacklist.txt``, and writes two artefacts:

* ``tetrad_motifs_by_pdb.json`` — a mapping ``"<pdb>-assembly<N>" -> [motif]``
  in the same shape as ``gnra_motifs_by_pdb.json`` so that step 02 can consume
  it with minimal changes.  Each ``unit_ids`` entry carries ``chain_id``,
  ``residue_number``, ``insertion_code`` and the ElTetrado ``fullName``.
* ``tetrad_exclusion_sets.json`` — per file, the set of ``(chain, number,
  icode)`` tuples that belong to an annotated G-tetrad.  Step 03 uses this to
  avoid generating negatives that overlap with real tetrads.

Both files are written into the ``tetrad/`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_JSON_DIR = "/mnt/data-ssd/tzok/onquadro-main/json"
DEFAULT_BLACKLIST = "/mnt/data-ssd/tzok/onquadro-main/blacklist.txt"
TETRAD_SIZE = 4


def load_blacklist(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


def is_g4_tetrad(tetrad: Dict[str, Any], short_name_map: Dict[str, str]) -> bool:
    """True iff all four nucleotides are guanines (RNA G or DNA DG)."""
    for key in ("nt1", "nt2", "nt3", "nt4"):
        full_name = tetrad[key]
        short = short_name_map.get(full_name)
        if short is None:
            # case-insensitive fallback
            short = next(
                (
                    v
                    for k, v in short_name_map.items()
                    if k.lower() == full_name.lower()
                ),
                None,
            )
        if short is None or short.upper() != "G":
            return False
    return True


def build_full_name_maps(nucleotides: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Return (fullName -> shortName, fullName -> nucleotide dict)."""
    short_map: Dict[str, str] = {}
    detail_map: Dict[str, Dict[str, Any]] = {}
    for nt in nucleotides:
        fn = nt.get("fullName")
        if fn is None:
            continue
        short_map[fn] = nt.get("shortName", "")
        detail_map[fn] = nt
    return short_map, detail_map


def residue_ref(full_name: str, detail_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resolve a tetrad ``nt1``..``nt4`` fullName to a chain/number/icode dict."""
    nt = detail_map.get(full_name)
    if nt is None:
        nt = next(
            (
                v
                for k, v in detail_map.items()
                if k.lower() == full_name.lower()
            ),
            None,
        )
    if nt is None:
        return None
    return {
        "chain_id": nt.get("chain"),
        "residue_number": nt.get("number"),
        "insertion_code": nt.get("icode") or "",
        "full_name": full_name,
    }


def extract_tetrads_from_file(
    json_path: str, short_map: Dict[str, str], detail_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int, str]]]:
    """Return (motifs, exclusion_residues) for one ElTetrado JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    motifs: List[Dict[str, Any]] = []
    exclusion: List[Tuple[str, int, str]] = []
    motif_index = 0

    for helix in data.get("helices", []):
        for quad in helix.get("quadruplexes", []):
            for tetrad in quad.get("tetrads", []):
                if not is_g4_tetrad(tetrad, short_map):
                    continue
                unit_ids: List[Dict[str, Any]] = []
                for key in ("nt1", "nt2", "nt3", "nt4"):
                    ref = residue_ref(tetrad[key], detail_map)
                    if ref is None:
                        break
                    unit_ids.append(ref)
                if len(unit_ids) != TETRAD_SIZE:
                    continue
                file_stem = Path(json_path).stem
                motif_key = f"G4_{file_stem}_{motif_index:03d}"
                motifs.append(
                    {
                        "motif_key": motif_key,
                        "unit_ids": unit_ids,
                        "onz": tetrad.get("onz"),
                        "gba_classification": tetrad.get("gbaClassification"),
                        "tetrad_id": tetrad.get("id"),
                    }
                )
                for u in unit_ids:
                    exclusion.append(
                        (u["chain_id"], u["residue_number"], u["insertion_code"])
                    )
                motif_index += 1

    return motifs, exclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help=f"ElTetrado JSON directory (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--blacklist",
        default=DEFAULT_BLACKLIST,
        help=f"Blacklist file (default: {DEFAULT_BLACKLIST})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
        help="Output directory for the two JSON artefacts",
    )
    args = parser.parse_args()

    blacklist = load_blacklist(args.blacklist)
    json_dir = args.json_dir
    output_dir = Path(args.output_dir)

    json_files = sorted(
        f for f in os.listdir(json_dir) if f.endswith(".json")
    )
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    print(f"Blacklist: {len(blacklist)} PDB entries")

    motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exclusion_by_pdb: Dict[str, List[List[Any]]] = {}

    total_tetrads = 0
    files_with_g4 = 0

    for fn in json_files:
        pdb_id = fn.split("-")[0].lower()
        if pdb_id in blacklist:
            continue
        json_path = os.path.join(json_dir, fn)
        with open(json_path) as f:
            data = json.load(f)
        short_map, detail_map = build_full_name_maps(data.get("nucleotides", []))
        if not short_map:
            continue
        stem = fn.replace(".json", "")
        motifs, exclusion = extract_tetrads_from_file(json_path, short_map, detail_map)
        if motifs:
            motifs_by_pdb[stem] = motifs
            exclusion_by_pdb[stem] = [list(e) for e in exclusion]
            total_tetrads += len(motifs)
            files_with_g4 += 1

    motifs_path = output_dir / "tetrad_motifs_by_pdb.json"
    exclusion_path = output_dir / "tetrad_exclusion_sets.json"

    with open(motifs_path, "w") as f:
        json.dump(dict(motifs_by_pdb), f, indent=2)
    with open(exclusion_path, "w") as f:
        json.dump(exclusion_by_pdb, f, indent=2)

    print(f"\nExtracted {total_tetrads} G4 tetrads from {files_with_g4} files")
    print(f"Distinct PDB entries: {len({k.split('-')[0] for k in motifs_by_pdb})}")
    print(f"Saved motifs  -> {motifs_path}")
    print(f"Saved exclusion -> {exclusion_path}")

    # Per-file tetrad count distribution
    counts = [len(v) for v in motifs_by_pdb.values()]
    from collections import Counter

    dist = Counter(counts)
    print(f"Tetrads-per-file distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
