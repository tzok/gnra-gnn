#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""KD-tree based enumeration of spatially close N-tuples of nucleotides.

Used by:

* ``03-generate-negative.py`` — to sample N-tuples of nearby nucleotides that
  do *not* overlap with annotated G-tetrads (negative examples).
* ``06-run-inference.py`` — to enumerate candidate N-tuples of guanines
  (potential tetrads) in a query structure, with a coplanarity / distance
  pruning step before classification.

The enumeration strategy is the same in both cases: build a ``cKDTree`` over
the C1' coordinates of a (sub)set of residues, and for each residue query its
neighbours within a radius ``R``; from each neighbour set we form all
``C(k, N)`` N-tuples that contain the seed residue, deduplicate by residue
identity, and optionally apply geometric pruning.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


ResidueId = Tuple[str, int, str]  # (chain_id, residue_number, insertion_code)
Point = Tuple[float, float, float]


def build_kdtree(coords: Sequence[Point]) -> Tuple[cKDTree, np.ndarray]:
    """Build a ``cKDTree`` over the given coordinates.

    Returns the tree and a ``(N, 3)`` float array of the coordinates.
    """
    pts = np.asarray(coords, dtype=float)
    tree = cKDTree(pts)
    return tree, pts


def neighbour_lists(
    tree: cKDTree, radius: float, include_self: bool = True
) -> List[np.ndarray]:
    """For each point, return the indices of its neighbours within ``radius``.

    When ``include_self`` is ``True`` each result includes the query index
    itself (matching the KD-tree default).
    """
    results = tree.query_ball_tree(tree, r=radius)
    if not include_self:
        results = [np.array([j for j in r if j != i]) for i, r in enumerate(results)]
    else:
        results = [np.array(r) for r in results]
    return results


def enumerate_tuples(
    coords: Sequence[Point],
    n: int,
    radius: float,
    seed_exclusion: Optional[Sequence[ResidueId]] = None,
    residue_ids: Optional[Sequence[ResidueId]] = None,
    max_distance: Optional[float] = None,
    coplanarity_rmsd: Optional[float] = None,
    dedup: bool = True,
) -> List[Tuple[Tuple[int, ...], Tuple[Point, ...]]]:
    """Enumerate N-tuples of nearby points.

    Parameters
    ----------
    coords
        C1' coordinates of all candidate residues.
    n
        Tuple size (4 for a tetrad).
    radius
        Neighbourhood radius for the KD-tree query (Å).
    seed_exclusion
        Residue IDs that must not appear in any returned tuple (e.g. residues
        that belong to annotated tetrads when generating negatives).  Must be
        paired with ``residue_ids``.  May be ``None`` to disable exclusion.
    residue_ids
        Residue IDs parallel to ``coords``; required when ``seed_exclusion`` is
        given or when ``dedup`` is ``True``.
    max_distance
        If set, drop tuples whose maximum pairwise C1' distance exceeds this
        value (Å).  Cheap geometric pruning.
    coplanarity_rmsd
        If set, drop tuples whose RMSD of the four C1' points to their
        best-fit plane exceeds this value (Å).  Tetrad-specific pruning.
    dedup
        If ``True``, remove tuples that are permutations of each other (same
        set of residues), keeping the canonical-order representative.

    Returns
    -------
    list of ``(indices, ordered_coords)``
        ``indices`` are the indices into ``coords`` in canonical order;
        ``ordered_coords`` are the corresponding points in canonical order.
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) < n:
        return []

    tree, _ = build_kdtree(pts)
    nbrs = tree.query_ball_tree(tree, r=radius)

    exclusion_set = set(seed_exclusion) if seed_exclusion is not None else None
    if residue_ids is None:
        residue_ids = [(f"c{i}", i, "") for i in range(len(pts))]

    seen_keys: set = set()
    tuples: List[Tuple[Tuple[int, ...], Tuple[Point, ...]]] = []

    for i, neigh in enumerate(nbrs):
        if len(neigh) < n:
            continue
        # Always include i; tuples that don't contain i will be produced when
        # their own seed is reached.  This keeps the per-iteration workload
        # bounded.
        neigh = sorted(neigh)
        for combo in combinations(neigh, n):
            if i not in combo:
                continue
            if exclusion_set is not None:
                if any(residue_ids[j] in exclusion_set for j in combo):
                    continue
            if max_distance is not None:
                sub = pts[list(combo)]
                dmax = float(np.max(np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)))
                if dmax > max_distance:
                    continue
            if coplanarity_rmsd is not None and n >= 3:
                sub = pts[list(combo)]
                if _plane_rmsd(sub) > coplanarity_rmsd:
                    continue
            if dedup:
                key = frozenset(residue_ids[j] for j in combo)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            # Canonical order
            from canonical_order import canonicalize

            sub_pts = [tuple(pts[j]) for j in combo]
            perm, ordered = canonicalize(sub_pts)
            ordered_indices = tuple(combo[k] for k in perm)
            tuples.append((ordered_indices, tuple(ordered)))

    return tuples


def _plane_rmsd(points: np.ndarray) -> float:
    """RMSD of ``points`` to their best-fit plane."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    # smallest eigenvalue of covariance = sum of squared distances to plane
    cov = np.cov(centred, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    # eigvalsh returns ascending; smallest eigenvalue = variance along normal
    # = mean of squared distances to plane (since centroid-subtracted)
    sum_sq = float(eigenvalues[0]) * len(points)
    return float(np.sqrt(max(0.0, sum_sq / len(points))))


def sample_negatives(
    coords: Sequence[Point],
    residue_ids: Sequence[ResidueId],
    n: int,
    radius: float,
    exclusion: Sequence[ResidueId],
    target_count: int,
    rng: np.random.Generator,
    max_distance: Optional[float] = None,
) -> List[Tuple[Tuple[int, ...], Tuple[Point, ...]]]:
    """Sample up to ``target_count`` negative N-tuples.

    Enumerates all candidate tuples (deduplicated, excluding ``exclusion``
    residues) and then randomly samples ``target_count`` of them.  When the
    candidate pool is smaller than ``target_count`` all of it is returned.
    """
    candidates = enumerate_tuples(
        coords,
        n=n,
        radius=radius,
        seed_exclusion=exclusion,
        residue_ids=residue_ids,
        max_distance=max_distance,
        dedup=True,
    )
    if len(candidates) <= target_count:
        return candidates
    idx = rng.choice(len(candidates), size=target_count, replace=False)
    return [candidates[i] for i in idx]
