# Double G-Tetrad (G-Quadruplex) Analysis Pipeline

This subdirectory applies the C1'-geometric-feature methodology to a
**two-tetrad G-quadruplex** — two consecutive planar G-tetrads stacked along a
quadruplex helix (8 guanines total).  It is the successor to the single-tetrad
pipeline in `tetrad/`, which hit a fundamental geometric degeneracy (documented
in `tetrad/README.md` under "Known limitation").  The 8-point geometry encodes
the vertical stacking and resolves that degeneracy.

## Why two tetrads (N=8) instead of one (N=4)?

The single-tetrad pipeline (`tetrad/`) demonstrated that 4 C1' points are
**information-theoretically insufficient** to distinguish a real G-tetrad
(horizontal square) from a "chimera" — a vertical rectangle mixing guanines
from two different stacking levels that are 3 levels apart.  The in-plane
G-G distance (~11.3 Å, Hoogsteen geometry) coincidentally equals 3 × the
rise per tetrad level (~3.7 Å, base stacking), so a horizontal square and a
vertical rectangle have nearly identical side lengths, diagonals, and hence
identical distance/angle/torsion features.  No model can learn the
difference from these features alone.

With 8 C1' points (two stacked tetrads), the feature space is 280-dimensional
and encodes the vertical stacking geometry.  A real double tetrad and a
chimera double tetrad (e.g. levels 2+5 instead of 2+3) produce feature vectors
that differ by up to 6 Å in individual distances — easily distinguishable.

## Relationship to the GNRA pipeline

This pipeline has the same feature-space size as the GNRA experiment
(280 features for N=8) and the same model-bundle format.  The key difference
is that the motif is **non-contiguous and multi-strand**: the 8 guanines come
from 4 different chains, not from a sliding sequence window.  Candidate
generation therefore uses KD-tree enumeration (as in `tetrad/`) rather than
a sliding window (as in the GNRA root pipeline).

## Source of annotations

Same as `tetrad/`: ElTetrado / OnQuadro (`/mnt/data-ssd/tzok/onquadro-main/`).
Step 01 finds **consecutive pairs** of G4 tetrads within each annotated
quadruplex.  A pair is kept only when both tetrads are G4 (all four
nucleotides are guanines, RNA or DNA).  This yields **1672 double tetrads**
from 723 files (590 distinct PDB entries).

## Pipeline steps

| Step | Script | Output | Description |
|------|--------|--------|-------------|
| 01 | `01-extract-double-tetrads.py` | `double_tetrad_motifs_by_pdb.json`, `double_tetrad_exclusion_sets.json` | Extract consecutive G4 tetrad pairs from ElTetrado JSON. |
| 02 | `02-generate-positive.py` | `motif_cif_files/DT_*.cif` | 8 G + canonicalisation → one CIF per double tetrad. |
| —  | `distiller` + `strip-distiller-json.py` | `approximate-*.json` (×3, slim) | Approximate-mode clustering of the 1672 positives. |
| 03 | `03-generate-negative.py` | `negative_cif_files/NEG_*.cif` | KD-tree 8-tuples with tuple-level exclusion (max_shared=6). |
| 04 | `04-generate-csv.py` | `geometric_features.csv` | 280 geometric features per sample. |
| 05 | `05-cluster-and-train.ipynb` | `*.pkl` (15 bundles) | 5 classifiers × 3 splits. |
| 06 | `06-run-inference.py` | `*-predictions.csv` (+ optional PNG) | KD-tree 8-tuples over guanines → predictions. |

Shared helpers: `canonical_order.py` (with height tiebreaker for stacked
tetrads), `features.py`, `neighborhood.py`.

## Running the pipeline

```bash
# 01 — extract double tetrads
uv run python double_tetrad/01-extract-double-tetrads.py

# 02 — write per-double-tetrad CIFs
uv run python double_tetrad/02-generate-positive.py --workers 8

# Clustering — approximate mode only
for method in hierarchical affinity-propagation facility-location; do
  uv run distiller --mode approximate --method $method \
    --output-json double_tetrad/approximate-${method}.json \
    double_tetrad/motif_cif_files/*.cif
done
uv run python double_tetrad/strip-distiller-json.py

# 03 — negatives
uv run python double_tetrad/03-generate-negative.py --workers 8

# 04 — feature CSV
uv run python double_tetrad/04-generate-csv.py

# 05 — train
uv run jupyter nbconvert --to notebook --execute double_tetrad/05-cluster-and-train.ipynb

# 06 — inference
uv run python double_tetrad/06-run-inference.py <structure.cif> \
    --model double_tetrad/approximate-hierarchical-random-forest.pkl \
    --output-csv double_tetrad/predictions.csv
```

## Parameters

* `--radius 22.0` — KD-tree neighbourhood radius (must span two stacked tetrads).
* `--max-distance 22.0` — max pairwise C1' distance within a candidate.
* `--k 5` — negatives multiplier.
* `--max-shared-with-tetrad 6` — tuple-level exclusion threshold (7/8 or 8/8
  shared with any annotated double tetrad → ignored; 0–6/8 → negative).

## Differences from `tetrad/` (N=4)

| Aspect | tetrad/ (N=4) | double_tetrad/ (N=8) |
|--------|---------------|----------------------|
| Motif | 1 G-tetrad (4 Gs) | 2 consecutive G-tetrads (8 Gs) |
| Features | 16 | 280 |
| Degeneracy | Yes (3-level chimeras) | No |
| Coplanarity pruning | Yes (1.5 Å) | No (8 pts span 2 planes) |
| Positives | 2541 | 1672 |
| Canonical order | angle sort | angle sort + **height tiebreaker** |
| Tuple exclusion threshold | max_shared=2 | max_shared=6 |
| Inference pruning | radius + coplanarity | radius only |
