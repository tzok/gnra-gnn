#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Dict, List, Any


def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


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


if __name__ == "__main__":
    main()
