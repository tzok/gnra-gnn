#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Any
from rnapolis.parser_v2 import parse_cif_atoms


def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def parse_mmcif_files(gnra_motifs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Parse mmCIF files for each PDB ID and return parsed structures."""
    parsed_structures = {}

    for pdb_id in gnra_motifs.keys():
        mmcif_file = f"mmcif_files/{pdb_id}.cif"

        if os.path.exists(mmcif_file):
            try:
                print(f"Parsing {mmcif_file}...")
                parsed_structure = parse_cif_atoms(mmcif_file)
                parsed_structures[pdb_id] = parsed_structure
                print(f"  Successfully parsed {pdb_id}")
            except Exception as e:
                print(f"  Error parsing {pdb_id}: {e}")
        else:
            print(f"  Warning: {mmcif_file} not found")

    return parsed_structures


def main():
    """Main function to parse GNRA motifs."""
    gnra_motifs = load_gnra_motifs()

    print(f"Loaded GNRA motifs for {len(gnra_motifs)} PDB structures")

    # Print summary information
    total_motifs = sum(len(motifs) for motifs in gnra_motifs.values())
    print(f"Total number of GNRA motifs: {total_motifs}")

    # Show first few PDB IDs and their motif counts
    for i, (pdb_id, motifs) in enumerate(gnra_motifs.items()):
        if i < 5:  # Show first 5
            print(f"  {pdb_id}: {len(motifs)} motifs")
        elif i == 5:
            print("  ...")
            break

    print("\nParsing mmCIF files...")
    parsed_structures = parse_mmcif_files(gnra_motifs)

    print(f"\nSuccessfully parsed {len(parsed_structures)} structures")


if __name__ == "__main__":
    main()
