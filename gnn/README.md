# GNN Model for RNA Motif Classification

This subdirectory contains the Graph Neural Network (GNN) code contributed by
Jan Pielesiak (PR #5), cleaned up from the original fork to remove generated
artefacts, duplicate files, and binary models.

## Overview

The GNN treats each RNA motif instance as a graph: nucleotides are nodes (with
one-hot encoded identity + geometric features), and pairwise distances /
angles / torsions are edge attributes.  The architecture is a 3-layer
GATv2 (Graph Attention Network v2) with global mean pooling, trained with
Optuna hyperparameter search.

Earlier GCN and standalone-GAT scripts were removed — they are superseded by
the notebook, which incorporates all leakage-prevention fixes (v6–v10):
temporal split, per-split cluster de-duplication, validation-based early
stopping, and label-shuffle sanity checks.

## Files

| File | Description |
|------|-------------|
| `01-prepare-mmcif.py` | Prepare mmCIF files for geometric feature extraction. |
| `02-generate-coordinates.py` | Extract 3D C1' coordinates + sequences from CIF files into CSV for GNN input. |
| `03-filter-by-date.py` | Split dataset by PDB release date (temporal holdout). |
| `04-train-gat.ipynb` | **Main training notebook** — GATv2 with Optuna, per-split dedup, temporal split, sanity checks. |
| `05-run-inference.py` | Inference script — supports both classical (sklearn) and GAT model bundles. |

## Dependencies

GNN-specific dependencies (torch, torch-geometric, imbalanced-learn, optuna)
are optional.  Install with:

```bash
uv sync --extra gnn
```

## Which file to use for training

**`04-train-gat.ipynb`** is the only training entry point.  It is
self-contained (all imports are external libraries) and includes every fix
from the v6→v10 evolution.  The earlier standalone scripts (`08-gnn.py`,
`08_gat.py`) were removed to avoid confusion.

## Relationship to other pipelines

This code was originally developed for the GNRA motif (8-nt contiguous window).
To adapt it for the `double_tetrad/` pipeline (8 guanines, non-contiguous,
multi-strand), the key changes are:

1. **Data source**: read from `double_tetrad/motif_cif_files/` and
   `negative_cif_files/` instead of the root-level directories.
2. **Graph topology**: replace `is_consecutive` edge flag with `is_same_strand`
   and `is_same_level` (two guanines from the same tetrad level are connected
   differently than two from the same strand).
3. **Inference**: replace the sliding-window candidate generation in
   `05-run-inference.py` with KD-tree enumeration from
   `double_tetrad/neighborhood.py` (8-tuples of nearby guanines).
4. **Feature columns**: already 280 for N=8 — identical format.

The `05-run-inference.py` `GATPredictor` wrapper is already sklearn-compatible,
so only the candidate-generation layer needs to change.
