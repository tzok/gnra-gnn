#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometric canonization of N C1' points into a deterministic order.

Tetrads (and arbitrary negative tuples) have no natural linear ordering the way
a contiguous sequence does.  The geometric features downstream are indexed
(``d01`` != ``d02``), so the same set of points fed in different orders would
yield different feature vectors.  This module returns a canonical permutation
of N 3D points that is invariant to the input order, translation, rotation, and
reflection (handedness is fixed to a single convention).

The algorithm:

1. Centre the points on their centroid.
2. PCA over the centred coordinates; the eigenvector with the smallest
   eigenvalue is the plane normal, the two largest eigenvectors span the
   best-fit plane.
3. Project the centred points onto the plane and compute their polar angle
   via ``atan2``.  Sorting these angles gives a cyclic order.
4. Fix handedness so the cyclic order is always counter-clockwise (positive
   signed area) when viewed along the normal; if not, reverse.
5. Choose the rotation start ``p0`` as the one whose centred coordinate tuple
   is lexicographically smallest — this makes the result independent of which
   angle happens to be first after the sort.

The same function is used for positives (step 02), negatives (step 03) and
inference candidates (step 06), guaranteeing that a given tetrad geometry
always produces the same feature vector.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float, float]


def _centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def _plane_basis(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis1, axis2, normal) of the best-fit plane through ``coords``.

    ``axis1`` corresponds to the largest eigenvalue, ``axis2`` to the second
    largest, ``normal`` to the smallest.  For perfectly planar input (e.g. a
    G-tetrad) the normal is still well defined up to sign; the sign is pinned
    later by the handedness step.
    """
    centred = coords - _centroid(coords)
    # covariance is 3x3, symmetric
    cov = np.cov(centred, rowvar=False)
    # eigh returns ascending eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # largest -> smallest variance
    normal = eigenvectors[:, 0]
    axis2 = eigenvectors[:, 1]
    axis1 = eigenvectors[:, 2]
    return axis1, axis2, normal


def canonicalize(coords: Sequence[Point]) -> Tuple[List[int], List[Point]]:
    """Return a canonical (permutation, ordered_coords) for ``coords``.

    Parameters
    ----------
    coords
        Iterable of N 3D points (tuples / lists / (N,3) array).

    Returns
    -------
    (permutation, ordered_coords)
        ``permutation[i]`` is the index in the input that ends up at position
        ``i`` in the canonical order.  ``ordered_coords`` are the corresponding
        input points in canonical order (not re-centred, so absolute
        coordinates are preserved for downstream distance / angle math).
    """
    pts = np.asarray(coords, dtype=float)
    n = pts.shape[0]
    if n < 3:
        # For fewer than 3 points there is no plane; fall back to input order
        # but still return a stable result sorted by coordinate tuple.
        order = sorted(range(n), key=lambda i: tuple(pts[i]))
        return order, [tuple(pts[i]) for i in order]

    axis1, axis2, normal = _plane_basis(pts)
    centred = pts - _centroid(pts)
    # 2D projection onto the best-fit plane
    u = centred @ axis1
    v = centred @ axis2
    angles = np.arctan2(v, u)

    # Cyclic order by angle
    cyclic = list(np.argsort(angles))

    # Fix handedness: signed area of the 2D polygon (u, v) in cyclic order.
    # Positive => counter-clockwise.  If negative, reverse to enforce CCW.
    poly_u = u[cyclic]
    poly_v = v[cyclic]
    signed_area = 0.5 * float(
        np.sum(
            poly_u * np.roll(poly_v, -1) - np.roll(poly_u, -1) * poly_v
        )
    )
    if signed_area < 0:
        cyclic = list(reversed(cyclic))

    # Choose rotation start p0 = lexicographically smallest centred point
    # among the cyclically-ordered points.  This decouples the result from the
    # arbitrary direction of the PCA eigenvectors.
    centred_tuples = [tuple(float(x) for x in centred[i]) for i in cyclic]
    start_offset = min(range(n), key=lambda k: centred_tuples[k])
    rotated = cyclic[start_offset:] + cyclic[:start_offset]

    ordered = [tuple(float(x) for x in pts[i]) for i in rotated]
    return rotated, ordered


def canonical_permutation_key(coords: Sequence[Point]) -> Tuple[Tuple[float, ...], ...]:
    """Return a hashable, order-independent key for de-duplication.

    Two point sets that canonize to the same key are the same tetrad up to
    rigid motion.  We use the sorted set of centred, canonical-ordered
    coordinates; because ``canonicalize`` already fixes order, rotation start
    and handedness, the key is just the resulting ordered centred coordinates.
    """
    pts = np.asarray(coords, dtype=float)
    _, ordered = canonicalize(pts)
    ordered_centred = np.array(ordered) - _centroid(pts)
    # Round to 3 decimals to be robust to FP noise
    return tuple(tuple(np.round(p, 3)) for p in ordered_centred)


def residue_tuple_key(residue_ids: Sequence[Tuple[str, int, str]]) -> frozenset:
    """Order-independent key for a set of residues (for exclusion / dedup).

    ``residue_ids`` are ``(chain_id, residue_number, insertion_code)`` tuples.
    """
    return frozenset(residue_ids)
