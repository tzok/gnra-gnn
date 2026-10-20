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
| 02 | `02-generate-positive.py` | `motif_cif_files/DT_*.cif` | 8 G in **topological order** → one CIF per double tetrad. |
| —  | `07-run-clustering.sh` | `approximate-*.json` (×3, slim) | Approximate-mode clustering of the 1672 positives (runs `distiller` + `strip-distiller-json.py`). |
| 03 | `03-generate-negative.py` | `negative_cif_files/NEG_*.cif` | KD-tree 8-tuples with tuple-level exclusion (max_shared=6). |
| 04 | `04-generate-csv.py` | `geometric_features.csv` | 280 geometric features per sample + **phase augmentation** + **shuffled hard negatives**. |
| 05 | `05-cluster-and-train.ipynb` | `*.pkl` (15 bundles) | 5 classifiers × 3 splits. |
| 06 | `06-run-inference.py` | `*-predictions.csv` (+ optional PNG) | KD-tree 8-tuples over guanines → predictions. |

Shared helpers: `topological_order.py` (chemistry-anchored ordering of
positives), `canonical_order.py` (geometric canonizer for negatives and
inference candidates, with height tiebreaker for stacked tetrads), `features.py`,
`neighborhood.py`.

## Residue ordering — topological, not geometric

The 280 features are **indexed** (`d01` ≠ `d02`), so the order in which the 8
residues are written to the CIF determines what each feature means.  The
original pipeline used a geometric canonizer (`canonical_order.canonicalize`)
for everything.  That is deterministic per point set but **not consistent
across structures**: its rotation-start tiebreak and PCA axis signs are
anchored to the crystal frame, so topologically identical double tetrads
arrived in different cyclic phases (20 distinct column sequences across the
1672 positives, none matching the intended convention) and identical features
meant different physical pairs from file to file — pure noise for the models.

Positives are now written in a **topological order** derived from the
ElTetrado annotation (`topological_order.py`):

1. ElTetrado's `nt1..nt4` is a cyclic walk around each tetrad (verified: walk
   neighbours sit at in-plane side distance, ~0.7× the diagonal), so motif
   `unit_ids = [lo_1..lo_4, hi_1..hi_4]` gives columns: column *i* =
   `(unit_ids[i], unit_ids[4+i])`.
2. The walk direction is normalised against the stacking normal
   (`centroid(upper) − centroid(lower)`) via the signed area of the lower
   tetrad's C1' polygon — relative geometry, hence invariant to rigid motion.
3. Column 1 is the column whose lower nucleotide has the smallest
   `(chain, resnum, icode)` identifier.

Result: `[lo_1, hi_1, lo_2, hi_2, lo_3, hi_3, lo_4, hi_4]`.  `d01` is the
intra-column stacking distance (5.3 ± 1.0 Å), `d02` the in-plane side
(11.5 ± 0.2 Å) — the same physical pair in every file.

## Phase augmentation and shuffled hard negatives (step 04)

At inference there is no annotation, so candidates (step 03 / step 06) are
still ordered by the geometric canonizer, whose cyclic phase depends on the
crystal frame.  To make the models phase-tolerant, step 04 augments each
positive with **every cyclic phase the canonizer can emit** (rigid random
rotations → re-canonize → deduplicate; typically 10–14 variants per motif,
all labelled positive), and adds **shuffled hard negatives** — random
permutations of true double tetrads whose index structure is not a valid
phase (1 per positive by default, `--shuffle-per-positive`).  Rows carry
`example_type` (`positive` / `phase_positive` / `geometric_negative` /
`shuffled_negative`) and `base_source_file`; augmented rows inherit the base
motif's `source_file` so the cluster-based holdout keeps all variants of a
motif in the same split.  Because augmentation makes the set positive-heavy,
the notebook trains with `class_weight="balanced"`.

## Running the pipeline

```bash
# 01 — extract double tetrads
uv run python double_tetrad/01-extract-double-tetrads.py

# 02 — write per-double-tetrad CIFs
uv run python double_tetrad/02-generate-positive.py --workers 8

# Clustering — approximate mode only (skips methods whose JSON exists; --force to redo)
uv run bash double_tetrad/07-run-clustering.sh

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
* `04-generate-csv.py`: `--shuffle-per-positive` (default 1, shuffled hard
  negatives per positive), `--no-augment` (skip phase variants and shuffles),
  `--seed`, `--max-rotation-samples` / `--rotation-patience` (phase-family
  sampling budget).

## Differences from `tetrad/` (N=4)

| Aspect | tetrad/ (N=4) | double_tetrad/ (N=8) |
|--------|---------------|----------------------|
| Motif | 1 G-tetrad (4 Gs) | 2 consecutive G-tetrads (8 Gs) |
| Features | 16 | 280 |
| Degeneracy | Yes (3-level chimeras) | No |
| Coplanarity pruning | Yes (1.5 Å) | No (8 pts span 2 planes) |
| Positives | 2541 | 1672 |
| Positives order | geometric canonizer | **topological** (columns + stacking normal + ID tiebreak) |
| Tuple exclusion threshold | max_shared=2 | max_shared=6 |
| Inference pruning | radius + coplanarity | radius only |
