#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 04 — compute geometric features for every positive / negative CIF.

Reads the per-tetrad CIF files from ``tetrad/motif_cif_files/`` (positives,
label ``tetrad = True``) and ``tetrad/negative_cif_files/`` (negatives, label
``tetrad = False``), extracts the four C1' atoms from each, computes the
16 geometric features (6 distances + 4·(sin,cos) planar angles +
1·(sin,cos) torsion) via ``features.calculate_geometric_features``, and writes
``tetrad/geometric_features.csv``.

This is the direct analogue of ``06-generate-csv.py`` from the GNRA pipeline,
generalised to ``N = 4``.  Raw planar-angle and torsion columns are kept in
the CSV so that the training notebook can drop them and keep only the
trigonometric transforms, exactly as in the GNRA experiment.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms

from features import calculate_geometric_features

TETRAD_SIZE = 4


def c1_prime_coords_from_cif(path: str) -> Optional[List[Tuple[float, float, float]]]:
    """Parse a CIF file and return the C1' coordinates in file order.

    Returns ``None`` if the file does not contain exactly ``TETRAD_SIZE`` C1'
    atoms (after de-duplicating within a residue).
    """
    try:
        with open(path) as f:
            atoms_df = parse_cif_atoms(f)
    except Exception as e:
        print(f"  Error parsing {path}: {e}")
        return None

    c1 = atoms_df[atoms_df["auth_atom_id"] == "C1'"]
    c1 = c1.drop_duplicates(
        subset=["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"], keep="first"
    )
    if len(c1) != TETRAD_SIZE:
        print(f"  {os.path.basename(path)}: {len(c1)} C1' atoms (expected {TETRAD_SIZE})")
        return None

    coords: List[Tuple[float, float, float]] = []
    for _, row in c1.iterrows():
        coords.append((float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])))
    return coords


def featurise_directory(directory: str, label: bool) -> List[pd.DataFrame]:
    """Return one single-row DataFrame per valid CIF in ``directory``."""
    files = sorted(glob.glob(os.path.join(directory, "*.cif")))
    print(f"Found {len(files)} .cif files in {directory}")
    rows: List[pd.DataFrame] = []
    for fp in files:
        coords = c1_prime_coords_from_cif(fp)
        if coords is None:
            continue
        feats = calculate_geometric_features(coords, n=TETRAD_SIZE)
        feats["source_file"] = Path(fp).stem
        feats["tetrad"] = label
        rows.append(pd.DataFrame([feats]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-dir",
        default=str(Path(__file__).parent / "motif_cif_files"),
    )
    parser.add_argument(
        "--negative-dir",
        default=str(Path(__file__).parent / "negative_cif_files"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "geometric_features.csv"),
    )
    args = parser.parse_args()

    print("Processing positive examples...")
    pos_rows = featurise_directory(args.positive_dir, label=True)
    print("\nProcessing negative examples...")
    neg_rows = featurise_directory(args.negative_dir, label=False)

    all_rows = pos_rows + neg_rows
    if not all_rows:
        print("No valid samples found.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} samples ({len(pos_rows)} positive, {len(neg_rows)} negative) to {args.output}")
    print(f"Total columns: {len(df.columns)} (1 source_file + 1 label + {len(df.columns) - 2} features)")


if __name__ == "__main__":
    main()
