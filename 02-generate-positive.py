#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import gzip
from typing import Dict, List, Any, Optional
from rnapolis.parser_v2 import parse_cif_atoms
from rnapolis.tertiary_v2 import Structure, Residue


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
    
    for motif in motifs:
        indices = []
        for unit_id_dict in motif:
            # Find matching residue by comparing unit_id components
            for i, residue in enumerate(residues):
                if (
                    residue.chain_id == unit_id_dict.get("chain_id")
                    and residue.number == unit_id_dict.get("residue_number")
                    and residue.insertion_code == unit_id_dict.get("insertion_code", "")
                ):
                    indices.append(i)
                    break
        motif_indices.append(indices)
    
    return motif_indices


def process_structures_and_motifs(
    structures: Dict[str, Structure], gnra_motifs: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[List[int]]]:
    """Process structures to find motif residue indices."""
    pdb_motif_indices = {}
    
    for pdb_id, structure in structures.items():
        print(f"Processing {pdb_id}...")
        residues = structure.residues()
        motifs = gnra_motifs[pdb_id]
        
        motif_indices = find_motif_residue_indices(residues, motifs)
        pdb_motif_indices[pdb_id] = motif_indices
        
        print(f"  Found {len(residues)} residues")
        print(f"  Processed {len(motifs)} motifs")
        for i, indices in enumerate(motif_indices):
            print(f"    Motif {i+1}: {len(indices)} residues at indices {indices}")
    
    return pdb_motif_indices


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
    structures = parse_mmcif_files(gnra_motifs)

    print(f"\nSuccessfully parsed {len(structures)} structures")
    
    print("\nProcessing structures and motifs...")
    pdb_motif_indices = process_structures_and_motifs(structures, gnra_motifs)
    
    print(f"\nProcessed motifs for {len(pdb_motif_indices)} PDB structures")


if __name__ == "__main__":
    main()
