#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Set

from rnapolis.parser_v2 import parse_cif_atoms
from rnapolis.tertiary_v2 import Residue, Structure


def load_gnra_motifs(
    filename: str = "gnra_motifs_by_pdb.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load GNRA motifs from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def load_structure_json(pdb_id: str) -> Dict[str, Any]:
    """Load structure JSON file for a PDB ID."""
    json_file = f"json_files/{pdb_id}.json"

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r") as f:
        return json.load(f)


def parse_and_process_mmcif_file(
    pdb_id: str, motifs: List[Dict[str, Any]]
) -> tuple[bool, Dict[str, Any]]:
    """Parse mmCIF file for a PDB ID and process its motifs."""
    mmcif_file = f"mmcif_files/{pdb_id}.cif"

    if not os.path.exists(mmcif_file):
        print(f"  Warning: {mmcif_file} not found")
        return False, {}

    try:
        print(f"Parsing {mmcif_file}...")
        with open(mmcif_file, "r") as f:
            atoms_df = parse_cif_atoms(f)
        structure = Structure(atoms_df)
        print(f"  Successfully parsed {pdb_id}")

        # Process motifs to find their indices
        residues = [residue for residue in structure.residues if residue.is_nucleotide]
        motif_data = find_motif_residue_indices(residues, motifs)

        print(f"  Found {len(residues)} residues")
        print(f"  Processed {len(motifs)} motifs")

        # Get GNRA motif indices (extended to 8 residues)
        gnra_indices = set()
        for motif_dict in motif_data:
            gnra_indices.update(motif_dict["indices"])

        # Load and analyze structure JSON
        try:
            structure_data = load_structure_json(pdb_id)
            negative_regions = find_negative_regions(structure_data, gnra_indices)

            print("  Found negative regions:")
            print(f"    Stems: {len(negative_regions['stems'])}")
            print(f"    Single strands: {len(negative_regions['single_strands'])}")
            print(f"    Hairpins: {len(negative_regions['hairpins'])}")
            print(f"    Loops: {len(negative_regions['loops'])}")

            # Add PDB ID to each region for tracking
            pdb_negative_regions = {"pdb_id": pdb_id, "regions": negative_regions}

            return True, pdb_negative_regions

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            return False, {}

    except Exception as e:
        print(f"  Error parsing {pdb_id}: {e}")
        return False, {}


def find_motif_residue_indices(
    residues: List[Residue], motifs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Find residue indices and residue objects for each motif's unit_ids, extending to 8 residues."""
    motif_data = []

    for motif_idx, motif in enumerate(motifs):
        indices: List[int] = []
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


def get_region_indices(region: Dict[str, Any]) -> List[int]:
    """Extract 1-based indices from a region (stem, single_strand, hairpin, or loop)."""
    indices: List[int] = []

    if "strand5p" in region and "strand3p" in region:
        # Stem region
        strand5p = region["strand5p"]
        strand3p = region["strand3p"]
        indices.extend(range(strand5p["first"], strand5p["last"] + 1))
        indices.extend(range(strand3p["first"], strand3p["last"] + 1))
    elif "strand" in region:
        # Single strand or hairpin
        strand = region["strand"]
        indices.extend(range(strand["first"], strand["last"] + 1))
    elif "strands" in region:
        # Loop with multiple strands
        for strand in region["strands"]:
            indices.extend(range(strand["first"], strand["last"] + 1))

    return sorted(set(indices))


def indices_overlap(indices1: List[int], indices2: Set[int]) -> bool:
    """Check if any indices from indices1 overlap with indices2."""
    return any(idx in indices2 for idx in indices1)


def find_negative_regions(
    structure_data: Dict[str, Any], gnra_indices: Set[int]
) -> Dict[str, List[Dict[str, Any]]]:
    """Find stems, single_strands, hairpins, and loops with at least 8 nucleotides that don't overlap with GNRA motifs."""
    negative_regions: Dict[str, List[Dict[str, Any]]] = {
        "stems": [],
        "single_strands": [],
        "hairpins": [],
        "loops": [],
    }

    # Convert 0-based GNRA indices to 1-based for comparison with bpseq indices
    gnra_indices_1based = {idx + 1 for idx in gnra_indices}

    # Process stems
    for stem in structure_data.get("stems", []):
        region_indices = get_region_indices(stem)
        if len(region_indices) >= 8 and not indices_overlap(
            region_indices, gnra_indices_1based
        ):
            negative_regions["stems"].append(
                {"region": stem, "indices": region_indices, "type": "stem"}
            )

    # Process single strands
    for single_strand in structure_data.get("single_strands", []):
        region_indices = get_region_indices(single_strand)
        if len(region_indices) >= 8 and not indices_overlap(
            region_indices, gnra_indices_1based
        ):
            negative_regions["single_strands"].append(
                {
                    "region": single_strand,
                    "indices": region_indices,
                    "type": "single_strand",
                }
            )

    # Process hairpins
    for hairpin in structure_data.get("hairpins", []):
        region_indices = get_region_indices(hairpin)
        if len(region_indices) >= 8 and not indices_overlap(
            region_indices, gnra_indices_1based
        ):
            negative_regions["hairpins"].append(
                {"region": hairpin, "indices": region_indices, "type": "hairpin"}
            )

    # Process loops
    for loop in structure_data.get("loops", []):
        region_indices = get_region_indices(loop)
        if len(region_indices) >= 8 and not indices_overlap(
            region_indices, gnra_indices_1based
        ):
            negative_regions["loops"].append(
                {"region": loop, "indices": region_indices, "type": "loop"}
            )

    return negative_regions


def process_pdb_wrapper(args):
    """Wrapper function for parallel processing."""
    pdb_id, motifs = args
    success, negative_regions = parse_and_process_mmcif_file(pdb_id, motifs)
    return pdb_id, success, negative_regions


def process_all_pdb_files(
    gnra_motifs: Dict[str, List[Dict[str, Any]]], max_workers: int = None
) -> List[Dict[str, Any]]:
    """Process all PDB files and their motifs in parallel."""
    successful_count = 0
    failed_count = 0
    all_negative_regions = []

    # Determine number of workers (default to number of CPU cores)
    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"Processing {len(gnra_motifs)} PDB files using {max_workers} workers...")

    # Prepare arguments for parallel processing
    pdb_args = list(gnra_motifs.items())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdb = {
            executor.submit(process_pdb_wrapper, args): args[0] for args in pdb_args
        }

        # Process completed tasks
        for future in as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                _, success, negative_regions = future.result()
                if success:
                    successful_count += 1
                    if negative_regions:
                        all_negative_regions.append(negative_regions)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"  Error processing {pdb_id}: {e}")
                failed_count += 1

    print("\nProcessing complete:")
    print(f"  Successfully processed: {successful_count} PDB files")
    print(f"  Failed to process: {failed_count} PDB files")

    return all_negative_regions


def save_negative_regions(
    negative_regions: List[Dict[str, Any]], filename: str = "negative_regions.json"
) -> None:
    """Save all negative regions to a JSON file."""
    with open(filename, "w") as f:
        json.dump(negative_regions, f, indent=2)
    print(
        f"Saved {len(negative_regions)} PDB structures with negative regions to {filename}"
    )


def load_negative_regions(
    filename: str = "negative_regions.json",
) -> List[Dict[str, Any]]:
    """Load negative regions from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def check_negative_regions_exist(filename: str = "negative_regions.json") -> bool:
    """Check if negative regions file already exists."""
    return os.path.exists(filename)


def print_negative_regions_summary(negative_regions: List[Dict[str, Any]]) -> None:
    """Print summary statistics of negative regions."""
    total_stems = 0
    total_single_strands = 0
    total_hairpins = 0
    total_loops = 0

    for pdb_data in negative_regions:
        regions = pdb_data.get("regions", {})
        total_stems += len(regions.get("stems", []))
        total_single_strands += len(regions.get("single_strands", []))
        total_hairpins += len(regions.get("hairpins", []))
        total_loops += len(regions.get("loops", []))

    print(f"\nNegative regions summary:")
    print(f"  Total PDB structures: {len(negative_regions)}")
    print(f"  Total stems: {total_stems}")
    print(f"  Total single strands: {total_single_strands}")
    print(f"  Total hairpins: {total_hairpins}")
    print(f"  Total loops: {total_loops}")
    print(
        f"  Total negative regions: {total_stems + total_single_strands + total_hairpins + total_loops}"
    )


def main():
    """Main function to find negative regions for GNRA motifs."""
    negative_regions_file = "negative_regions.json"

    # Check if negative regions file already exists
    if check_negative_regions_exist(negative_regions_file):
        print(f"Negative regions file '{negative_regions_file}' already exists.")
        print("Loading existing negative regions...")
        negative_regions = load_negative_regions(negative_regions_file)
        print_negative_regions_summary(negative_regions)
        print("\nTo regenerate, delete the file and run the script again.")
        return

    gnra_motifs = load_gnra_motifs()

    print(f"Loaded GNRA motifs for {len(gnra_motifs)} PDB structures")

    # Print summary information
    total_motifs = sum(len(motifs) for motifs in gnra_motifs.values())
    print(f"Total number of GNRA motifs: {total_motifs}")

    print("\nAnalyzing structures for negative regions...")
    negative_regions = process_all_pdb_files(gnra_motifs)

    # Save negative regions to file
    save_negative_regions(negative_regions, negative_regions_file)

    # Print summary
    print_negative_regions_summary(negative_regions)


if __name__ == "__main__":
    main()
