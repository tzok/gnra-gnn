#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 03 — generate negative examples (non-tetrad 4-tuples) via KD-tree.

For every (pdb, assembly) stem that contributed positive tetrads, this script:

1. Parses the assembly mmCIF and collects every nucleotide residue (RNA *or*
   DNA) together with its C1' coordinates.
2. Builds a ``scipy.spatial.cKDTree`` over the C1' points.
3. Enumerates all 4-tuples of nucleotides whose C1' atoms fall within a
   radius ``R`` of each other (see ``neighborhood.enumerate_tuples``), with
   per-tuple geometric pruning (max pairwise distance, coplanarity RMSD).
4. Excludes any tuple that contains a residue belonging to an annotated
   G-tetrad of the same assembly (the exclusion set comes from
   ``tetrad_exclusion_sets.json`` produced by step 01).
5. Randomly samples up to ``K * <positive count in this file>`` tuples
   (deduplicated by residue set), canonicalises their order, and writes one
   CIF file per sampled tuple into ``tetrad/negative_cif_files/``.

Defaults are tuned to the geometry of a G-tetrad (C1'-C1' side ~11.5 Å,
diagonal ~16.3 Å): ``R = 18 Å``, ``max_distance = 18 Å``,
``coplanarity_rmsd = 1.5 Å``, ``K = 5``.  All are CLI-overridable.

The model only ever sees C1' geometry, so by mixing RNA and DNA nucleotides
in the negatives we keep the negative distribution aligned with the positive
one (which also mixes RNA and DNA tetrads) and avoid introducing a
sequence / molecule-type shortcut.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure

from neighborhood import enumerate_tuples

DEFAULT_MIRROR = "/mnt/data-ssd/tzok/onquadro-main/mirror/data/assemblies/mmCIF/divided"
TETRAD_SIZE = 4
DEFAULT_RADIUS = 18.0
DEFAULT_MAX_DISTANCE = 18.0
DEFAULT_COPLANARITY_RMSD = 1.5
DEFAULT_K = 5
DEFAULT_SEED = 42


def mmcif_path(stem: str, mirror_dir: str) -> str:
    """Return the gzipped mmCIF path for ``<pdb>-assembly<N>``."""
    pdb_id = stem.split("-")[0]
    mid = pdb_id[1:3]
    return os.path.join(mirror_dir, mid, f"{stem}.cif.gz")


def collect_nucleotides(
    structure: Structure,
) -> Tuple[List[Residue], List[Tuple[float, float, float]], List[Tuple[str, int, str]]]:
    """Return parallel lists of (residues, c1' coords, residue IDs)."""
    residues: List[Residue] = []
    coords: List[Tuple[float, float, float]] = []
    ids: List[Tuple[str, int, str]] = []
    for r in structure.residues:
        if not r.is_nucleotide:
            continue
        col_atom = r._col("atom_name")
        c1 = r.atoms[r.atoms[col_atom] == "C1'"]
        if c1.empty:
            continue
        row = c1.iloc[0]
        residues.append(r)
        coords.append(
            (float(row[r._col("x")]), float(row[r._col("y")]), float(row[r._col("z")]))
        )
        ids.append((r.chain_id, r.residue_number, r.insertion_code or ""))
    return residues, coords, ids


def process_one_stem(
    stem: str,
    exclusion: List[Tuple[str, int, str]],
    positive_count: int,
    mirror_dir: str,
    output_dir: str,
    radius: float,
    max_distance: Optional[float],
    coplanarity_rmsd: Optional[float],
    k: int,
    seed: int,
) -> Tuple[str, int, int]:
    """Generate negatives for one (pdb, assembly) stem.

    Returns ``(stem, n_saved, n_skipped)``.
    """
    path = mmcif_path(stem, mirror_dir)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping {stem}")
        return stem, 0, 0

    try:
        with gzip.open(path, "rt") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
    except Exception as e:
        print(f"  Error parsing {stem}: {e}")
        return stem, 0, 0

    residues, coords, ids = collect_nucleotides(structure)
    if len(residues) < TETRAD_SIZE:
        return stem, 0, 0

    exclusion_set = set(map(tuple, exclusion))

    candidates = enumerate_tuples(
        coords,
        n=TETRAD_SIZE,
        radius=radius,
        seed_exclusion=exclusion_set,
        residue_ids=ids,
        max_distance=max_distance,
        coplanarity_rmsd=coplanarity_rmsd,
        dedup=True,
    )

    if not candidates:
        return stem, 0, 0

    target = k * positive_count
    rng = np.random.default_rng(seed)
    if len(candidates) > target:
        idx = rng.choice(len(candidates), size=target, replace=False)
        sampled = [candidates[i] for i in idx]
    else:
        sampled = candidates

    n_saved = 0
    n_skipped = 0
    for j, (indices, _ordered) in enumerate(sampled):
        out_file = os.path.join(output_dir, f"NEG_{stem}_{j:04d}.cif")
        if os.path.exists(out_file):
            n_skipped += 1
            continue
        # `indices` are already in canonical order from enumerate_tuples,
        # so the residues written to the CIF are in canonical order too.
        ordered_residues = [residues[i] for i in indices]
        try:
            atoms_df = pd.concat(r.atoms for r in ordered_residues)
            with open(out_file, "w") as f:
                write_cif(atoms_df, f)
            n_saved += 1
        except Exception as e:
            print(f"    Error saving NEG_{stem}_{j:04d}: {e}")
            n_skipped += 1

    return stem, n_saved, n_skipped


def process_wrapper(args):
    return process_one_stem(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motifs",
        default=str(Path(__file__).parent / "tetrad_motifs_by_pdb.json"),
        help="Path to tetrad_motifs_by_pdb.json (for positive counts per stem)",
    )
    parser.add_argument(
        "--exclusion",
        default=str(Path(__file__).parent / "tetrad_exclusion_sets.json"),
        help="Path to tetrad_exclusion_sets.json",
    )
    parser.add_argument(
        "--mirror",
        default=DEFAULT_MIRROR,
        help=f"OnQuadro mmCIF mirror directory (default: {DEFAULT_MIRROR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "negative_cif_files"),
        help="Output directory for negative CIF files",
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--coplanarity-rmsd", type=float, default=DEFAULT_COPLANARITY_RMSD)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.motifs) as f:
        motifs_by_pdb: Dict[str, List[Dict[str, Any]]] = json.load(f)
    with open(args.exclusion) as f:
        exclusion_by_pdb: Dict[str, List[List[Any]]] = json.load(f)

    print(f"Loaded {sum(len(v) for v in motifs_by_pdb.values())} positives across {len(motifs_by_pdb)} stems")
    print(
        f"Params: radius={args.radius} Å, max_distance={args.max_distance} Å, "
        f"coplanarity_rmsd={args.coplanarity_rmsd} Å, K={args.k}, seed={args.seed}"
    )

    tasks = []
    for stem, motifs in motifs_by_pdb.items():
        exclusion = [tuple(e) for e in exclusion_by_pdb.get(stem, [])]
        tasks.append(
            (
                stem,
                exclusion,
                len(motifs),
                args.mirror,
                output_dir,
                args.radius,
                args.max_distance,
                args.coplanarity_rmsd,
                args.k,
                args.seed + hash(stem) % (2**31),
            )
        )

    total_saved = 0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_wrapper, t): t[0] for t in tasks}
        for future in as_completed(futures):
            stem = futures[future]
            try:
                _, saved, skipped = future.result()
                total_saved += saved
                total_skipped += skipped
                if saved:
                    print(f"  {stem}: saved {saved} negatives, skipped {skipped}")
            except Exception as e:
                print(f"  Error processing {stem}: {e}")

    print(f"\nDone. Saved {total_saved} negatives, skipped {total_skipped}.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
