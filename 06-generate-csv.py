#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
from pathlib import Path

import pandas as pd
from rnapolis.parser_v2 import parse_cif_atoms


def process_cif_files_for_c1_prime(directory: str) -> pd.DataFrame:
    """
    Process all *.cif files in a directory, extract C1' atoms, and return a dataframe.

    Only includes files that have exactly 8 C1' atoms.

    Args:
        directory: Path to directory containing .cif files

    Returns:
        DataFrame with C1' atoms and a 'source_file' column indicating the origin file
    """
    all_dataframes = []

    # Find all .cif files in the directory
    cif_pattern = os.path.join(directory, "*.cif")
    cif_files = glob.glob(cif_pattern)

    print(f"Found {len(cif_files)} .cif files in {directory}")

    for cif_file in cif_files:
        try:
            # Parse the CIF file
            with open(cif_file, "r") as fd:
                atoms_df = parse_cif_atoms(fd)

            # Filter for C1' atoms only
            c1_prime_atoms = atoms_df[atoms_df["auth_atom_id"] == "C1'"]

            # Check if we have exactly 8 C1' atoms
            if len(c1_prime_atoms) == 8:
                # Add source file column
                filename = Path(cif_file).stem  # Get filename without extension
                c1_prime_atoms = c1_prime_atoms.copy()
                c1_prime_atoms["source_file"] = filename

                all_dataframes.append(c1_prime_atoms)
                print(f"  ✓ {filename}: Found exactly 8 C1' atoms")
            else:
                filename = Path(cif_file).stem
                print(
                    f"  ✗ {filename}: Found {len(c1_prime_atoms)} C1' atoms (expected 8)"
                )

        except Exception as e:
            filename = Path(cif_file).stem
            print(f"  ✗ {filename}: Error parsing file - {e}")

    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(
            f"\nSuccessfully processed {len(all_dataframes)} files with exactly 8 C1' atoms"
        )
        print(f"Total C1' atoms in combined dataframe: {len(combined_df)}")
        return combined_df
    else:
        print("\nNo files with exactly 8 C1' atoms found")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    directory = "motif_cif_files"  # Replace with actual directory path
    df = process_cif_files_for_c1_prime(directory)

    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Source files: {df['source_file'].unique()}")
