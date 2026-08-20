# G4 Tetrad Analysis Pipeline

This subdirectory applies the **same C1'-geometric-feature methodology** used
for the GNRA tetraloop (in the parent directory) to a different, non-contiguous
RNA / DNA motif: the **G-tetrad** — the planar quartet of guanines that is the
building block of a G-quadruplex.  The goal is to demonstrate that a model
trained purely on C1' geometry (no sequence, no base atoms, no hydrogen-bond
annotations) can recognise tetrads in arbitrary structures.

The original GNRA pipeline works on a contiguous 8-nt window, so a sliding
sequence window suffices for both training-data extraction and inference.  A
G-tetrad, by contrast, draws its four guanines from up to four different
strands and is never contiguous in sequence, so the pipeline is rebuilt around
**KD-tree enumeration of spatially close guanine quartets** rather than a
sliding window.  Everything else (geometric feature definitions, clustering
for redundancy removal, classical-classifier training, bundled-model format)
is kept deliberately parallel to the GNRA experiment.

## Why a separate subdirectory

The GNRA scripts (`01`–`11` in the project root) are a finished, working
experiment and are left untouched.  All tetrad-specific scripts live under
`tetrad/` and share the project's `uv` environment (the only added dependency
is `scipy`, for the KD-tree).

## Motif: single G-tetrad (4 nt, not 8)

A **single** G-tetrad is used (not a two-tetrad quadruplex) for two reasons:

1. It gives a feature-space size different from the GNRA experiment
   (16 features for N=4 vs 280 for N=8), which makes the demonstration that
   *the C1' approach is flexible with respect to motif size* stronger.
2. A two-tetrad (8-nt) quadruplex would simply reproduce the 280-feature setup
   of the GNRA pipeline and obscure the comparison.

For N = 4 the geometric feature vector contains:

* `C(4,2) = 6` pairwise C1'–C1' distances,
* `C(4,3) = 4` planar angles, each encoded as (sin, cos) → 8 features,
* `C(4,4) = 1` torsion angle, encoded as (sin, cos) → 2 features,

**16 features** total (raw `a`/`t` columns are dropped before training, just
as in the GNRA notebook).

## Source of annotations: ElTetrado / OnQuadro

Positive examples come from the local OnQuadro mirror produced by ElTetrado
(`/mnt/data-ssd/tzok/onquadro-main/`):

* `json/<pdb>-assembly<N>.json` — ElTetrado analysis.  Tetrads are listed under
  `helices[].quadruplexes[].tetrads[]` with `nt1`..`nt4` (ElTetrado `fullName`
  like `A.G2` or `A.DG3`).
* `mirror/data/assemblies/mmCIF/divided/<XX>/<pdb>-assembly<N>.cif.gz` — the
  gzipped source mmCIF files (`<XX>` is the middle two characters of the PDB
  ID).
* `blacklist.txt` — PDB IDs to skip (20 entries).

A tetrad is kept as a positive iff all four nucleotides are guanines.  Because
ElTetrado reports `shortName` per nucleotide, the filter uses `shortName == "G"`
which matches both RNA `G` and DNA `DG`.  **DNA tetrads are intentionally
included** — this increases the positive pool from 535 (RNA only) to **2541
G-tetrads** across 595 distinct PDB entries and 16 ElTetrado GBA geometric
classes, removing the small-sample risk and forcing the model to learn
geometry rather than a single conformational template.  Because the model only
ever sees C1' coordinates, mixing RNA and DNA introduces no shortcut.

## The canonical-order problem

A contiguous-sequence motif (GNRA) has a natural linear order; a tetrad does
not.  Geometric features are index-dependent (`d01 ≠ d02`), so the same four
points fed in different orders produce different feature vectors.  The module
`canonical_order.py` resolves this with a deterministic, input-order-invariant
canonicalisation:

1. Centre the points on their centroid.
2. PCA → the smallest-eigenvalue eigenvector is the plane normal, the two
   largest span the best-fit plane.
3. Project onto the plane and compute each point's polar angle (`atan2`).
4. Sort by angle to get a cyclic order.
5. Fix handedness to counter-clockwise (positive signed area); reverse if
   needed.
6. Rotate the cyclic order so that it starts at the lexicographically
   smallest centred point — this makes the result independent of the PCA
   eigenvectors' arbitrary signs.

The same `canonicalize` is used for positives (step 02), negatives (step 03)
and inference candidates (step 06), so any given tetrad geometry always
yields the same feature vector.

## Negative generation: tuple-level exclusion (partial-ignore)

A G-quadruplex structure typically contains **all** of its guanines in some
annotated tetrad.  A naïve residue-level exclusion (drop any candidate
containing *any* residue from *any* tetrad) therefore eliminates every
4-guanine tuple from the negative set — including "chimeras" that mix
guanines from two different stacking levels (e.g. G2, G2, G5, G5).  These
chimeras have nearly-identical C1' geometry to real tetrads (same square
arrangement, same ~11.5 Å side, ~16 Å diagonal, coplanar) yet are *not*
tetrads.  Without them as hard negatives, the model cannot learn to
distinguish them and produces false positives at inference time.

Step 03 instead uses **tuple-level exclusion** with a partial-ignore rule,
directly analogous to "ignore regions" in object detection (YOLO, Faster
R-CNN), where ambiguous overlap cases are discarded rather than forced into
positive or negative:

| Candidate shares with any annotated tetrad | Treatment | Rationale |
|----------------------------------------------|-----------|-----------|
| 4 / 4 residues | **ignore** (it is the positive itself) | already in the positive set |
| 3 / 4 residues | **ignore** (ambiguous, "almost a tetrad") | forcing it negative would be methodologically questionable; a reviewer could argue it is a borderline case |
| 0 – 2 / 4 residues | **negative** | includes 2+2 chimeras (two residues from one tetrad level, two from another) — the hardest negatives, and the ones the model must learn to reject |

The threshold (`--max-shared-with-tetrad`, default 2) is configurable.  This
keeps the negative set free of both positives and ambiguous near-positives,
while retaining the informative hard negatives that a residue-level exclusion
would have discarded.

## Known limitation: geometric degeneracy of single-tetrad features

Despite the partial-ignore exclusion strategy, the single-tetrad (N=4) model
cannot reliably distinguish real G-tetrads from **chimera** tuples that mix
guanines from two different stacking levels 3 levels apart (e.g. G2, G2, G5,
G5).  This is an **information-theoretic limit** of the 4-point C1' feature
space, not a training issue:

* In-plane adjacent G-G distance (Hoogsteen geometry): **~11.3 Å**
* Rise per tetrad level (base stacking): **~3.7 Å**
* 3 levels × 3.7 ≈ 11.1 Å ≈ 11.3 Å

A horizontal square (real tetrad) and a vertical rectangle (chimera) therefore
have nearly identical side lengths and diagonals.  Because distances, planar
angles, and torsion angles are all **rotation-invariant** by construction, the
two configurations produce nearly identical 16-feature vectors (max difference
~0.3 Å).  No classifier — regardless of training set size, negative quality,
or algorithm — can learn to separate them.

This was confirmed empirically: on structure 1j8g, the model produces false
positives on 2+2 chimeras (levels 2 and 5) with probabilities ≥ 0.97,
indistinguishable from real tetrads.

**Resolution:** the `double_tetrad/` pipeline (N=8, two consecutive tetrads)
addresses this by encoding the vertical stacking geometry.  With 8 C1' points
the feature space is 280-dimensional, and real double tetrads vs chimeras
differ by up to 6 Å in individual distances — easily distinguishable.  See
`double_tetrad/README.md` for details.

## Pipeline steps

| Step | Script | Output | Description |
|------|--------|--------|-------------|
| 01 | `01-extract-g4-tetrads.py` | `tetrad_motifs_by_pdb.json`, `tetrad_exclusion_sets.json` | Filter ElTetrado JSON for G-tetrads (G + DG), apply blacklist, emit per-(pdb,assembly) motif list + exclusion residue sets. |
| 02 | `02-generate-positive.py` | `motif_cif_files/G4_*.cif` | For each annotated tetrad, open the assembly mmCIF, locate the four guanines, canonicalise their order, write one CIF per tetrad. |
| —  | `distiller` (external, run by user) + `strip-distiller-json.py` | `approximate-*.json` (×3 methods, slim) | Cluster the 2541 positive CIFs by geometric redundancy (PCA-based approximate mode), then strip diagnostics so only `clustering.clusters` remains (a few hundred KB total). Same JSON shape as the GNRA clustering. The `exact` mode is omitted because all-vs-all nRMSD over 2541 structures is computationally infeasible; the approximate mode is sufficient for redundancy removal. |
| 03 | `03-generate-negative.py` | `negative_cif_files/NEG_*.cif` | For each stem, KD-tree over all nucleotide C1' atoms, enumerate spatially close 4-tuples (radius 18 Å), prune by coplanarity & max distance, apply **tuple-level exclusion** (ignore candidates sharing >2 residues with any annotated tetrad — see below), sample `K × #positives` per file. |
| 04 | `04-generate-csv.py` | `geometric_features.csv` | Compute the 16 geometric features + `source_file` + `tetrad` label for every positive and negative CIF. |
| 05 | `05-cluster-and-train.ipynb` | `*.pkl` (15 bundles) | Load the CSV, drop raw angle columns, build train/test splits per clustering variant, train 5 classifiers × 3 splits, pickle each as a self-contained bundle. |
| 06 | `06-run-inference.py` | `*-predictions.csv` (+ optional PNG) | On an arbitrary structure: collect guanines, KD-tree enumerate tetrad-shaped 4-tuples, canonicalise, featurise, classify, emit CSV with predictions + probabilities + residue metadata. |

Shared helper modules (not numbered steps):

* `canonical_order.py` — geometric canonisation of N points.
* `features.py` — N-atom geometric features (generalised from the GNRA
  8-atom version).
* `neighborhood.py` — `cKDTree`-based N-tuple enumeration with pruning and
  dedup.

## Running the pipeline

From the project root (`uv sync` first if needed):

```bash
# 01 — extract G-tetrads from ElTetrado JSON (a few seconds)
uv run python tetrad/01-extract-g4-tetrads.py

# 02 — write per-tetrad CIFs (parallel; ~1 minute for 2541 tetrads)
uv run python tetrad/02-generate-positive.py --workers 8

# Clustering — approximate mode only (run by user; minutes). The exact mode
# is skipped because all-vs-all nRMSD over 2541 CIFs is computationally
# infeasible. Produces approximate-{hierarchical,affinity-propagation,
# facility-location}.json inside tetrad/.
for method in hierarchical affinity-propagation facility-location; do
  uv run distiller --mode approximate --method $method \
    --output-json tetrad/approximate-${method}.json \
    tetrad/motif_cif_files/*.cif
done

# Strip distiller diagnostics so only clustering.clusters remains (~170 KB
# per file instead of up to ~650 MB). Must be run before committing.
uv run python tetrad/strip-distiller-json.py

# 03 — negative examples (parallel; sampling bounded by K=5 × positives/file)
uv run python tetrad/03-generate-negative.py --workers 8

# 04 — feature CSV
uv run python tetrad/04-generate-csv.py

# 05 — train models (notebook)
uv run jupyter nbconvert --to notebook --execute tetrad/05-cluster-and-train.ipynb

# 06 — inference on a structure
uv run python tetrad/06-run-inference.py <structure.cif> \
    --model tetrad/approximate-hierarchical-random-forest.pkl \
    --output-csv tetrad/predictions.csv --output-plot tetrad/probabilities.png
```

## Model bundle format

Identical to the GNRA pipeline's bundle, with `window_size = 4` instead of 8:

| Key               | Description                                             |
|-------------------|---------------------------------------------------------|
| `classifier_name` | Human-readable classifier name                          |
| `classifier`      | Fitted scikit-learn estimator                           |
| `scaler`          | Fitted `StandardScaler`                                 |
| `feature_columns` | List of 16 feature column names in training order       |
| `window_size`     | `4`                                                     |
| `positive_label`  | `True`                                                  |
| `split_mode`      | `"approximate"` (exact mode skipped — see Pipeline steps) |
| `split_method`    | `"hierarchical"`, `"affinity-propagation"`, ...        |

## Parameters (defaults tuned to G-tetrad geometry)

A G-tetrad's four C1' atoms form a square of side ≈ 11.5 Å and diagonal
≈ 16.3 Å, lying in a single plane.  The negative-generation and inference
defaults reflect this:

* `--radius 18.0` — KD-tree neighbourhood radius (must exceed the tetrad
  diagonal so a real tetrad's four Gs are all mutual neighbours).
* `--max-distance 18.0` — max pairwise C1' distance within a candidate tuple.
* `--coplanarity-rmsd 1.5` — max RMSD of the four C1' points to their
  best-fit plane (real tetrads are ~0; this is generous to accommodate
  imperfect crystallographic planarity).
* `--k 5` — negatives multiplier (per file: `5 × #positive tetrads` sampled).
* `--max-shared-with-tetrad 2` — tuple-level exclusion threshold: a candidate
  sharing more than this many residues with any annotated tetrad is ignored
  (4/4 = positive, 3/4 = ambiguous, 0–2/4 = negative).

All are CLI-overridable.

## Differences from the GNRA pipeline (summary)

| Aspect | GNRA (root) | G-tetrad (here) |
|--------|-------------|-----------------|
| Motif size | 8 nt (contiguous) | 4 nt (non-contiguous, multi-strand) |
| Features | 280 (28 d + 56·2 angles + 70·2 torsions) | 16 (6 d + 4·2 angles + 1·2 torsions) |
| Annotation source | FR3D atlas (`hl_3.97.json`) | ElTetrado / OnQuadro (`json/*.json`) |
| Molecule type | RNA only | RNA + DNA |
| Positive ordering | Sequence order | Geometric canonisation (PCA + cyclic + handedness) |
| Negative generation | Structural elements (stems/loops/hairpins) | KD-tree spatial neighbourhoods + **tuple-level partial-ignore exclusion** (hard 2+2 chimeras kept) |
| Inference candidates | 8-nt sliding sequence window | KD-tree 4-tuples over guanines (pruned) |
| Clustering | `distiller` external (exact + approximate) | `distiller` external, **approximate only** (exact infeasible at 2541 structures) |
| Training notebook | `10-remove-redundancy-and-train-models.ipynb` | `05-cluster-and-train.ipynb` (N=4, label `tetrad`) |
| Bundle `window_size` | 8 | 4 |

## Files

```
tetrad/
├── 01-extract-g4-tetrads.py      # ElTetrado JSON -> motif + exclusion JSON
├── 02-generate-positive.py       # 4 G + canonisation -> motif_cif_files/
├── 03-generate-negative.py       # KD-tree sampling -> negative_cif_files/
├── 04-generate-csv.py            # 16 features -> geometric_features.csv
├── 05-cluster-and-train.ipynb    # clustering splits -> 15 model bundles
├── 06-run-inference.py           # KD-tree over G -> predictions CSV + plot
├── strip-distiller-json.py       # shrink distiller output to clustering.clusters
├── canonical_order.py            # shared: geometric canonisation
├── features.py                   # shared: N-atom geometric features
├── neighborhood.py               # shared: KD-tree N-tuple enumeration
├── README.md                     # this file
├── tetrad_motifs_by_pdb.json     # (generated by 01)
├── tetrad_exclusion_sets.json    # (generated by 01)
├── motif_cif_files/              # (generated by 02)
├── negative_cif_files/           # (generated by 03)
├── geometric_features.csv        # (generated by 04)
├── approximate-*.json            # (generated by distiller)
└── *.pkl                         # (generated by 05)
```
