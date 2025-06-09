#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
from typing import Any, Dict, List

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
) -> List[List[int]]:
    """Find residue indices for each motif's unit_ids."""
    motif_indices = []

    for motif_idx, motif in enumerate(motifs):
        indices = []
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
                    break

        # Log when we don't find exactly 6 indices
        if len(indices) != 6:
            print(
                f"    Warning: {motif_key} - Expected 6 residues, found {len(indices)}: {indices}"
            )
            continue  # Skip adding this motif to motif_indices

        # Log when indices are not consecutive
        sorted_indices = sorted(indices)
        is_consecutive = all(
            sorted_indices[i] + 1 == sorted_indices[i + 1] for i in range(5)
        )
        if not is_consecutive:
            print(
                f"    Warning: {motif_key} - Residues are not consecutive: {sorted_indices}"
            )
            continue  # Skip adding this motif to motif_indices

        motif_indices.append(indices)

    return motif_indices


def extract_and_save_motif(
    structure: Structure, indices: List[int], motif_key: str
) -> bool:
    """Extract 8 residues (6 motif + 1 before + 1 after) and save as CIF file."""
    # Create output directory
    output_dir = "motif_cif_files"
    os.makedirs(output_dir, exist_ok=True)

    # Check if file already exists
    output_file = os.path.join(output_dir, f"{motif_key}.cif")
    if os.path.exists(output_file):
        print(f"    File {output_file} already exists, skipping")
        return False

    # Get the range of indices (add 1 before and 1 after)
    min_idx = min(indices)
    max_idx = max(indices)

    # Check if we can add residues before and after
    if min_idx == 0 or max_idx == len(structure.residues) - 1:
        print(f"    Cannot extract 8 residues for {motif_key} (boundary constraints)")
        return False

    # Extract 8 residues: 1 before + 6 motif + 1 after
    extended_indices = [min_idx - 1] + indices + [max_idx + 1]

    try:
        # Get atoms for the extended residues
        all_atom_indices = []
        for idx in extended_indices:
            residue = structure.residues[idx]
            # Use the atoms field from the residue to get atom indices
            atom_indices = [atom.index for atom in residue.atoms]
            all_atom_indices.extend(atom_indices)

        # Create a new dataframe with just these atoms
        atoms_df = structure.atoms.loc[all_atom_indices]

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
) -> Dict[str, List[List[int]]]:
    """Process structures to find motif residue indices."""
    pdb_motif_indices = {}

    for pdb_id, structure in structures.items():
        print(f"Processing {pdb_id}...")
        residues = structure.residues
        motifs = gnra_motifs[pdb_id]

        motif_indices = find_motif_residue_indices(residues, motifs)
        pdb_motif_indices[pdb_id] = motif_indices

        print(f"  Found {len(residues)} residues")
        print(f"  Processed {len(motifs)} motifs")

        # Process valid motifs and extract CIF files
        valid_motif_count = 0
        for motif, indices in zip(motifs, motif_indices):
            if indices:  # Only process motifs that passed validation
                valid_motif_count += 1
                motif_key = motif.get("motif_key", f"motif_{valid_motif_count}")
                print(
                    f"    Motif {valid_motif_count}: {len(indices)} residues at indices {indices}"
                )
                extract_and_save_motif(structure, indices, motif_key)

    return pdb_motif_indices


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
