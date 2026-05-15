# GNRA Motif Analysis Pipeline

This project provides a comprehensive pipeline for analyzing GNRA tetraloop motifs in RNA structures from the Protein Data Bank (PDB). The pipeline downloads structural data, extracts motifs, generates negative examples, performs machine learning classification using geometric features, and applies trained models to new structures.

## Overview

The pipeline consists of several scripts that work together to:

1. Download mmCIF files for PDB structures containing GNRA motifs
2. Extract and process individual motifs from the structures
3. Analyze base pairing patterns using FR3D
4. Generate negative examples from non-GNRA regions
5. Extract geometric features from C1' atoms
6. Train classifiers per clustering-based train/test split and export bundled model artifacts
7. Run a trained model on an arbitrary structure using 8-nt sliding windows

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

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install and sync dependencies:

```bash
uv sync
```

The main dependencies (declared in `pyproject.toml`) are:

- `rnapolis`: Library for parsing and manipulating RNA structures
- `scikit-learn`: Machine learning algorithms and evaluation metrics
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Plotting support for inference outputs
- `jupyter` / `ipykernel`: Running the training notebook

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

### 5. Generate Negative Examples

```bash
python 05-generate-negative.py
```

This script:

- Identifies structural elements that don't overlap with GNRA motifs
- Extracts negative examples from hairpin loops, internal loops, and single strands
- Ensures all examples are 8 nucleotides long and from the same chain
- Outputs negative example files to `negative_cif_files/` directory

### 6. Generate Geometric Features

```bash
python 06-generate-csv.py
```

This script:

- Processes all CIF files from both `motif_cif_files/` and `negative_cif_files/`
- Extracts C1' atoms and calculates geometric features:
  - Pairwise distances between all atoms
  - Planar angles for all triplets of atoms
  - Torsion angles for all quadruplets of atoms
  - Sine and cosine transformations of all angles
- Generates a comprehensive CSV file with labeled examples

### 7. Machine Learning Classification (deprecated)

`07-classical-ml.py` is **deprecated** in favor of the notebook below. It performed k-fold cross-validation with a neural network and is retained for reference only.

### 8. Train Models Using Clustering-Based Splits

Open and run the notebook:

```
jupyter notebook 10-remove-redundancy-and-train-models.ipynb
```

Or execute it non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute 10-remove-redundancy-and-train-models.ipynb
```

This notebook:

- Loads the geometric features dataset (`geometric_features.csv`)
- Drops redundant raw-angle columns (keeping only sine/cosine transformations)
- Creates train/test splits from per-method clustering results (`approximate-*.json`, `exact-*.json`)
- Trains five classifiers on each split:
  - Naive Bayes
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
- Pickles each classifier as a self-contained **bundled artifact** containing:
  - `classifier` — the fitted estimator
  - `scaler` — the `StandardScaler` fitted on the training set
  - `feature_columns` — the exact 280 feature columns in training order
  - `window_size`, `positive_label`, `split_mode`, `split_method` — metadata

This produces files such as `approximate-hierarchical-random-forest.pkl`, one per split and classifier combination.

### 9. Run a Trained Model on a Structure

```bash
uv run python 11-run-trained-model-on-structure.py <structure> --model <model.pkl>
```

Example:

```bash
uv run python 11-run-trained-model-on-structure.py mmcif_files/1g1x.cif --model approximate-hierarchical-random-forest.pkl
```

With an optional probability plot:

```bash
uv run python 11-run-trained-model-on-structure.py mmcif_files/1g1x.cif \
    --model approximate-hierarchical-random-forest.pkl \
    --output-plot out.png
```

This script:

- Parses the input structure (`.cif` or `.pdb`) using `rnapolis.parser_v2`
- Builds an 8-nucleotide moving window across every nucleic-acid chain
- Computes the same geometric features used during training (distances, sine/cosine angles, sine/cosine torsions)
- Reindexes features to match the bundled `feature_columns` order
- Applies the bundled scaler and classifier
- Writes a CSV with window metadata, predictions, and (when available) per-window positive-class probabilities
- Optionally generates a probability plot when `--output-plot` is provided

**Note:** SVM models must be trained with `SVC(probability=True)` to produce probability outputs; otherwise the plot step is skipped.

## File Structure

```
.
├── 01-download-cif.py                              # Download mmCIF files
├── 02-generate-positive.py                         # Extract positive examples (GNRA motifs)
├── 03-analyze-with-fr3d.sh                         # Run FR3D analysis
├── 04-extract-elements.sh                          # Extract structural elements
├── 05-generate-negative.py                         # Generate negative examples
├── 06-generate-csv.py                              # Extract geometric features
├── 07-classical-ml.py                              # (deprecated) k-fold CV classification
├── 10-remove-redundancy-and-train-models.ipynb     # Clustering-split model training (notebook)
├── 11-run-trained-model-on-structure.py            # Inference on arbitrary structures
├── pyproject.toml                                  # Project metadata and dependencies
├── uv.lock                                         # Locked dependency versions
├── gnra_motifs_by_pdb.json                         # Input motif data
├── geometric_features.csv                          # Extracted geometric features dataset
├── negative_regions.json                           # Generated negative regions data
├── approximate-{method}.json                       # Approximate clustering results
├── exact-{method}.json                             # Exact clustering results
├── mmcif_files/                                    # Downloaded mmCIF files
├── motif_cif_files/                                # Extracted GNRA motif files
├── negative_cif_files/                             # Extracted negative example files
└── json_files/                                     # Structure JSON analysis data
```

## Features

- **Parallel Processing**: Uses multiprocessing for efficient handling of large datasets
- **Resume Capability**: Automatically skips already processed files
- **Error Handling**: Robust error handling with informative logging
- **Memory Efficient**: Processes files individually rather than loading all into memory
- **Chain-Aware Processing**: Filters structural elements by chains containing GNRA motifs
- **Geometric Feature Extraction**: Comprehensive calculation of distances, angles, and torsions
- **Multiple ML Algorithms**: Comparison of Naive Bayes, Logistic Regression, Decision Tree, Random Forest, and SVM classifiers
- **Self-Contained Model Artifacts**: Each pickled model bundles its scaler and feature-column order, enabling standalone inference without re-deriving the training pipeline
- **Sliding-Window Inference**: Run any bundled model on whole structures via 8-nt scanning windows
- **Probability Plotting**: Optional per-window probability visualization for supported classifiers

## Model Bundle Format

Every `.pkl` file produced by `10-remove-redundancy-and-train-models.ipynb` is a Python dictionary with the following keys:

| Key               | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `classifier_name` | Human-readable classifier name (e.g. `"Random Forest"`) |
| `classifier`      | The fitted scikit-learn estimator                       |
| `scaler`          | The fitted `StandardScaler`                             |
| `feature_columns` | List of 280 feature column names in training order      |
| `window_size`     | Always `8`                                              |
| `positive_label`  | Always `True`                                           |
| `split_mode`      | Clustering mode (e.g. `"approximate"`, `"exact"`)       |
| `split_method`    | Clustering method (e.g. `"hierarchical"`)               |

The `11-run-trained-model-on-structure.py` script validates this format on load.

## Notes

- The pipeline is designed to handle large datasets efficiently
- All scripts include progress reporting and error handling
- Files are processed incrementally, allowing for interrupted runs to be resumed
- The FR3D analysis step requires the `cli2rest-bio` tool to be properly installed and configured
- Use `uv sync` for reproducible environments
- The `07-classical-ml.py` script is deprecated; use the notebook and inference script instead
- Geometric features include trigonometric transformations to handle angular periodicity
- Multi-model PDB/mmCIF files are handled correctly by the inference script (defaults to model 1 unless `--structure-model` is provided)
