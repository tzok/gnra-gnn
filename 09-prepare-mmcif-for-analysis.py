import glob
import importlib
import math
import os
from pathlib import Path
from typing import List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms

import importlib.util

spec = importlib.util.spec_from_file_location(
    "generate_csv_module",  # arbitrary name
    "06-generate-csv.py"
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def process_cif_files_for_c1_prime(directory: str,filepath: str = None) -> List[pd.DataFrame]:
    """
    Process all *.cif files in a directory, extract C1' atoms, and return a dataframe.

    Turns one mmcif file into a dataframe of each group of 8 C1' atoms.

    Args:
        directory: Path to directory containing .cif files
        filepath: Optional path to a single .cif file to process instead of the whole directory

    Returns:
        DataFrame with C1' atoms and a 'source_file' column indicating the origin file
    """
    all_dataframes = []
    # Find all .cif files in the directory
    if filepath:
        cif_files = [filepath]
    else:
        cif_pattern = os.path.join(directory, "*.cif")
        cif_files = sorted(glob.glob(cif_pattern))

    print(f"Found {len(cif_files)} .cif files in {directory}")

    for cif_file in cif_files:
        try:
            # Parse the CIF file
            with open(cif_file, "r") as fd:
                atoms_df = parse_cif_atoms(fd)

            # Filter for C1' atoms only
            c1_prime_atoms = atoms_df[atoms_df["auth_atom_id"] == "C1'"]

            # Remove duplicate C1' atoms within the same residue - keep only the first occurrence
            # Group by residue identifiers and take the first occurrence of each group
            c1_prime_atoms = c1_prime_atoms.drop_duplicates(
                subset=["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"],
                keep="first",
            )
            # Check if we have exactly 8 C1' atoms
            for i in range(0,len(c1_prime_atoms)-7):
                group_of_8 = c1_prime_atoms.iloc[i:i+8]
                # Add source file column
                filename = Path(cif_file).stem  # Get filename without extension
                group_of_8 = group_of_8.copy()
                group_of_8["source_file"] = f"{filename}_NT:{i}_{i+7}"

                all_dataframes.append(group_of_8)
                print(f" {filename}: Saving {i} {i+7} C1' atoms")


        except Exception as e:
            filename = Path(cif_file).stem
            print(f"  ✗ {filename}: Error parsing file - {e}")

    # Combine all dataframes
    if all_dataframes:
        print(
            f"\nSuccessfully processed {len(all_dataframes)} files with exactly 8 C1' atoms"
        )
        return all_dataframes
    else:
        print("\nNo files with exactly 8 C1' atoms found")
        return [pd.DataFrame()]
    

if __name__ == "__main__":
    # Process positive examples (GNRA motifs)
    print("Processing positive examples from motif_cif_files...")
    dfs = process_cif_files_for_c1_prime("test_1JID")

    features = []
    for df in dfs:
        if not df.empty:
            feats = module.calculate_geometric_features(df)
            feats["gnra"] = True
            features.append(feats)

    if features:
        # Concatenate all feature dataframes
        final_df = pd.concat(features, ignore_index=True)

        # Save to CSV
        output_file = "geometric_features_to_test4.csv"
        final_df.to_csv(output_file, index=False)

        print(f"\nSaved {len(final_df)} samples to {output_file}")
        print(f"Total features per sample: {len(final_df.columns)}")
    else:
        print("\nNo valid samples found to process")