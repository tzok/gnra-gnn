#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
from typing import Any, Dict, List

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms, write_cif
from rnapolis.tertiary_v2 import Residue, Structure


def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def parse_mmcif_files(
    gnra_motifs: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Structure]:
    """Parse mmCIF files for each PDB ID and return Structure objects."""
    structures = {}

    for pdb_id in gnra_motifs.keys():
        mmcif_file = f"mmcif_files/{pdb_id}.cif.gz"

        if os.path.exists(mmcif_file):
            try:
                print(f"Parsing {mmcif_file}...")
                with gzip.open(mmcif_file, "rt") as f:
                    atoms_df = parse_cif_atoms(f)
                structure = Structure(atoms_df)
                structures[pdb_id] = structure
                print(f"  Successfully parsed {pdb_id}")
            except Exception as e:
                print(f"  Error parsing {pdb_id}: {e}")
        else:
            print(f"  Warning: {mmcif_file} not found")

    return structures


def find_motif_residue_indices(
    residues: List[Residue], motifs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Find residue indices and residue objects for each motif's unit_ids, extending to 8 residues."""
    motif_data = []

    for motif_idx, motif in enumerate(motifs):
        indices = []
        motif_residues = []
        unit_ids = motif.get("unit_ids", [])
        motif_key = motif.get("motif_key", f"motif_{motif_idx}")

        for unit_id_dict in unit_ids:
            # Find matching residue by comparing unit_id components
            for i, residue in enumerate(residues):
                unit_insertion_code = unit_id_dict.get("insertion_code", "")
                residue_insertion_code = residue.insertion_code or ""

                if (
                    residue.chain_id == unit_id_dict.get("chain_id")
                    and residue.residue_number == unit_id_dict.get("residue_number")
                    and residue_insertion_code == unit_insertion_code
                ):
                    indices.append(i)
                    motif_residues.append(residue)
                    break

        # Log when we don't find exactly 6 indices
        if len(indices) != 6:
            print(
                f"    Warning: {motif_key} - Expected 6 residues, found {len(indices)}: {indices}"
            )
            continue  # Skip adding this motif to motif_data

        # Log when indices are not consecutive
        sorted_indices = sorted(indices)
        is_consecutive = all(
            sorted_indices[i] + 1 == sorted_indices[i + 1] for i in range(5)
        )
        if not is_consecutive:
            print(
                f"    Warning: {motif_key} - Residues are not consecutive: {sorted_indices}"
            )
            continue  # Skip adding this motif to motif_data

        # Extend to 8 residues (add 1 before and 1 after)
        min_idx = min(sorted_indices)
        max_idx = max(sorted_indices)

        # Check if we can add residues before and after
        if min_idx == 0 or max_idx == len(residues) - 1:
            print(
                f"    Warning: {motif_key} - Cannot extend to 8 residues (boundary constraints)"
            )
            continue  # Skip adding this motif to motif_data

        # Create extended indices and residues
        extended_indices = [min_idx - 1] + sorted_indices + [max_idx + 1]
        extended_residues = [residues[i] for i in extended_indices]

        motif_data.append(
            {
                "motif_key": motif_key,
                "indices": extended_indices,
                "residues": extended_residues,
            }
        )

    return motif_data


def extract_and_save_motif(
    motif_dict: Dict[str, Any],
) -> bool:
    """Extract 8 residues (6 motif + 1 before + 1 after) and save as CIF file."""
    motif_key = motif_dict["motif_key"]
    indices = motif_dict["indices"]
    residues = motif_dict["residues"]

    # Create output directory
    output_dir = "motif_cif_files"
    os.makedirs(output_dir, exist_ok=True)

    # Check if file already exists
    output_file = os.path.join(output_dir, f"{motif_key}.cif")
    if os.path.exists(output_file):
        print(f"    File {output_file} already exists, skipping")
        return False

    try:
        # Get atoms for the extended residues
        atoms_df = pd.concat(residue.atoms for residue in residues)

        # Write to CIF file
        with open(output_file, "w") as f:
            write_cif(atoms_df, f)

        print(f"    Saved {motif_key} to {output_file}")
        return True

    except Exception as e:
        print(f"    Error saving {motif_key}: {e}")
        return False


def process_structures_and_motifs(
    structures: Dict[str, Structure], gnra_motifs: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Process structures to find motif residue indices."""
    pdb_motif_data = {}

    for pdb_id, structure in structures.items():
        print(f"Processing {pdb_id}...")
        residues = [residue for residue in structure.residues if residue.is_nucleotide]
        motifs = gnra_motifs[pdb_id]

        motif_data = find_motif_residue_indices(residues, motifs)
        pdb_motif_data[pdb_id] = motif_data

        print(f"  Found {len(residues)} residues")
        print(f"  Processed {len(motifs)} motifs")

        # Process valid motifs and extract CIF files
        for motif_dict in motif_data:
            motif_key = motif_dict["motif_key"]
            indices = motif_dict["indices"]
            print(
                f"    Motif {motif_key}: {len(indices)} residues at indices {indices}"
            )
            extract_and_save_motif(motif_dict)

    return pdb_motif_data


def main():
    """Main function to parse GNRA motifs."""
    gnra_motifs = load_gnra_motifs()

    print(f"Loaded GNRA motifs for {len(gnra_motifs)} PDB structures")

    # Print summary information
    total_motifs = sum(len(motifs) for motifs in gnra_motifs.values())
    print(f"Total number of GNRA motifs: {total_motifs}")

    # TODO
    gnra_motifs = dict(list(gnra_motifs.items())[:5])  # Limit to first 5 for testing

    print("\nParsing mmCIF files...")
    structures = parse_mmcif_files(gnra_motifs)

    print(f"\nSuccessfully parsed {len(structures)} structures")

    print("\nProcessing structures and motifs...")
    pdb_motif_indices = process_structures_and_motifs(structures, gnra_motifs)

    print(f"\nProcessed motifs for {len(pdb_motif_indices)} PDB structures")


if __name__ == "__main__":
    main()
