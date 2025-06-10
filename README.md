# GNRA Motif Analysis Pipeline

This project provides a comprehensive pipeline for analyzing GNRA (GNRA tetraloop) motifs in RNA structures from the Protein Data Bank (PDB). The pipeline downloads structural data, extracts motifs, generates negative examples, and performs machine learning classification using geometric features.

## Overview

The pipeline consists of several scripts that work together to:

1. Download mmCIF files for PDB structures containing GNRA motifs
2. Extract and process individual motifs from the structures
3. Analyze base pairing patterns using FR3D
4. Generate negative examples from non-GNRA regions
5. Extract geometric features from C1' atoms
6. Train and evaluate machine learning classifiers

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
- `scikit-learn`: Machine learning algorithms and evaluation metrics
- `tensorflow`: Deep learning framework for neural networks
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing

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

### 7. Machine Learning Classification

```bash
python 07-classical-ml.py
```

This script:
- Loads the geometric features dataset
- Trains multiple classifiers:
  - Naive Bayes
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - Neural Network (TensorFlow/Keras)
- Evaluates performance using accuracy, precision, recall, and F1-score
- Provides detailed misclassification analysis

## File Structure

```
.
├── 01-download-cif.py          # Download mmCIF files
├── 02-generate-positive.py     # Extract positive examples (GNRA motifs)
├── 03-analyze-with-fr3d.sh     # Run FR3D analysis
├── 04-extract-elements.sh      # Extract structural elements
├── 05-generate-negative.py     # Generate negative examples
├── 06-generate-csv.py          # Extract geometric features
├── 07-classical-ml.py          # Machine learning classification
├── requirements.txt            # Python dependencies
├── gnra_motifs_by_pdb.json    # Input motif data
├── hl_3.97.json               # FR3D structural analysis data
├── negative_regions.json      # Generated negative regions data
├── geometric_features.csv     # Extracted geometric features dataset
├── mmcif_files/               # Downloaded mmCIF files
├── motif_cif_files/           # Extracted GNRA motif files
└── negative_cif_files/        # Extracted negative example files
```

## Features

- **Parallel Processing**: Uses multiprocessing for efficient handling of large datasets
- **Resume Capability**: Automatically skips already processed files
- **Error Handling**: Robust error handling with informative logging
- **Memory Efficient**: Processes files individually rather than loading all into memory
- **Chain-Aware Processing**: Filters structural elements by chains containing GNRA motifs
- **Geometric Feature Extraction**: Comprehensive calculation of distances, angles, and torsions
- **Multiple ML Algorithms**: Comparison of classical and deep learning approaches
- **Detailed Performance Analysis**: Comprehensive evaluation with multiple metrics

## Machine Learning Results

The pipeline generates a comprehensive comparison of different classification algorithms:

- **Classical Methods**: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, SVM
- **Deep Learning**: Neural Network with dropout and multiple hidden layers
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Misclassification Analysis**: Detailed reporting of incorrectly classified instances

### Current Performance Results

```
Classifier           Accuracy   Precision  Recall     F1-Score
----------------------------------------------------------------------
Naive Bayes          0.9934     0.9688     0.9841     0.9764
Logistic Regression  0.9934     0.9839     0.9683     0.9760
Neural Network       0.9934     0.9688     0.9841     0.9764
Random Forest        0.9913     0.9683     0.9683     0.9683
Decision Tree        0.9825     0.9365     0.9365     0.9365
SVM                  0.9803     0.8750     1.0000     0.9333
```

The results show excellent performance across all classifiers, with Naive Bayes, Logistic Regression, and Neural Network achieving the highest accuracy of 99.34%. The SVM achieves perfect recall (100%) but with lower precision, while other methods show more balanced precision-recall trade-offs.

## Notes

- The pipeline is designed to handle large datasets efficiently
- All scripts include progress reporting and error handling
- Files are processed incrementally, allowing for interrupted runs to be resumed
- The FR3D analysis step requires the `cli2rest-bio` tool to be properly installed and configured
- Neural network training uses early stopping and model checkpointing for optimal performance
- Geometric features include trigonometric transformations to handle angular periodicity
