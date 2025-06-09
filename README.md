# GNRA Motif Analysis Pipeline

This project provides a pipeline for analyzing GNRA (GNRA tetraloop) motifs in RNA structures from the Protein Data Bank (PDB). The pipeline downloads structural data, extracts motifs, and performs detailed analysis using FR3D.

## Overview

The pipeline consists of several scripts that work together to:

1. Download mmCIF files for PDB structures containing GNRA motifs
2. Extract and process individual motifs from the structures
3. Analyze base pairing patterns using FR3D
4. Generate detailed structural data for further analysis

## Prerequisites

### System Requirements

- **GNU Parallel**: Required for parallel processing of multiple files
  ```bash
  # On Ubuntu/Debian
  sudo apt-get install parallel
  
  # On macOS with Homebrew
  brew install parallel
  
  # On Arch Linux
  sudo pacman -S parallel
  ```

### Python Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `cli2rest-bio`: Interface to FR3D for RNA structure analysis
- `rnapolis`: Library for parsing and manipulating RNA structures

## Usage

### 1. Download mmCIF Files

```bash
python 01-download-cif.py
```

This script:
- Reads GNRA motif data from `gnra_motifs_by_pdb.json`
- Downloads mmCIF files for each PDB structure
- Stores uncompressed `.cif` files in `mmcif_files/` directory
- Skips files that already exist

### 2. Generate Positive Examples

```bash
python 02-generate-positive.py
```

This script:
- Processes mmCIF files in parallel for efficiency
- Extracts individual GNRA motifs and saves them as separate CIF files
- Skips PDB structures where all motifs are already processed
- Outputs motif files to `motif_cif_files/` directory

### 3. Analyze with FR3D

```bash
./03-analyze-with-fr3d.sh
```

This script:
- Checks which CIF files need FR3D analysis
- Runs FR3D analysis only on files without existing output
- Generates base pairing detail files (`fr3d-{pdb_id}-basepair_detail.txt`)

### 4. Extract Elements

```bash
./04-extract-elements.sh
```

Additional processing script for extracting specific structural elements.

## File Structure

```
.
├── 01-download-cif.py          # Download mmCIF files
├── 02-generate-positive.py     # Extract motifs
├── 03-analyze-with-fr3d.sh     # Run FR3D analysis
├── 04-extract-elements.sh      # Additional processing script
├── requirements.txt            # Python dependencies
├── gnra_motifs_by_pdb.json    # Input motif data
├── hl_3.97.json               # Additional input data
├── mmcif_files/               # Downloaded mmCIF files
└── motif_cif_files/           # Extracted motif files
```

## Features

- **Parallel Processing**: Uses multiprocessing for efficient handling of large datasets
- **Resume Capability**: Automatically skips already processed files
- **Error Handling**: Robust error handling with informative logging
- **Memory Efficient**: Processes files individually rather than loading all into memory

## Notes

- The pipeline is designed to handle large datasets efficiently
- All scripts include progress reporting and error handling
- Files are processed incrementally, allowing for interrupted runs to be resumed
- The FR3D analysis step requires the `cli2rest-bio` tool to be properly installed and configured
