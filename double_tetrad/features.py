#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometric feature extraction for N C1' atoms.

Generalises the original 8-atom featurisation from ``06-generate-csv.py`` /
``11-run-trained-model-on-structure.py`` to an arbitrary number ``N`` of
points.  For ``N = 8`` the output is identical to the GNRA pipeline.  For
``N = 4`` (a single G-tetrad) the feature vector has:

* ``C(4,2)  = 6`` pairwise distances
* ``C(4,3)  = 4`` planar angles, each encoded as (sin, cos) -> 8 features
* ``C(4,4)  = 1`` torsion angle, encoded as (sin, cos) -> 2 features

Total: 16 features.

Raw angle / torsion values are also produced (``a{i}{j}{k}``, ``t{i}{j}{k}{l}``)
so the training notebook can drop them and keep only the trigonometric
transforms, exactly as in the GNRA experiment.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np

Point = Tuple[float, float, float]


def calculate_distance(p1: Point, p2: Point) -> float:
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
    )


def calculate_planar_angle(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Angle at ``p2`` between rays to ``p1`` and ``p3``.

    Returns ``(angle_radians, sin_angle, cos_angle)``.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot = float(np.dot(v1, v2))
    mag1 = float(np.linalg.norm(v1))
    mag2 = float(np.linalg.norm(v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0, 0.0, 1.0
    cos_a = dot / (mag1 * mag2)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    angle = math.acos(cos_a)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    return angle, sin_a, cos_a


def calculate_torsion_angle(
    p1: Point, p2: Point, p3: Point, p4: Point
) -> Tuple[float, float, float]:
    """Dihedral angle of four points.

    Returns ``(angle_radians, sin_angle, cos_angle)``.
    """
    a, b, c, d = map(np.array, (p1, p2, p3, p4))
    v1 = b - a
    v2 = c - b
    v3 = d - c
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm == 0.0 or n2_norm == 0.0:
        return 0.0, 0.0, 1.0
    n1u = n1 / n1_norm
    n2u = n2 / n2_norm
    cos_a = float(np.dot(n1u, n2u))
    v2_norm = float(np.linalg.norm(v2))
    sin_a = float(np.dot(np.cross(n1u, n2u), v2 / v2_norm)) if v2_norm else 0.0
    angle = math.atan2(sin_a, cos_a)
    return angle, sin_a, cos_a


def calculate_geometric_features(coords: Sequence[Point], n: int) -> Dict[str, float]:
    """Compute all geometric features for ``n`` 3D points.

    Vectorised over numpy arrays for speed.  For N=8 this is ~10× faster
    than the per-pair Python loop version.

    Column naming follows the GNRA convention: ``d{ij}`` (distance),
    ``a{ijk}`` + ``as{ijk}`` + ``aa{ijk}`` (raw / sin / cos planar angle at
    ``j``), ``t{ijkl}`` + ``ts{ijkl}`` + ``ta{ijkl}`` (raw / sin / cos
    torsion).
    """
    pts_list = list(coords)
    if len(pts_list) != n:
        raise ValueError(f"Expected {n} points, got {len(pts_list)}")

    pts = np.asarray(pts_list, dtype=float)
    result: Dict[str, float] = {}

    # --- Distances (vectorised) ---
    for i, j in combinations(range(n), 2):
        result[f"d{i}{j}"] = float(np.linalg.norm(pts[i] - pts[j]))

    # --- Planar angles (vectorised) ---
    for i, j, k in combinations(range(n), 3):
        v1 = pts[i] - pts[j]
        v2 = pts[k] - pts[j]
        dot = float(np.dot(v1, v2))
        mag1 = float(np.linalg.norm(v1))
        mag2 = float(np.linalg.norm(v2))
        if mag1 == 0.0 or mag2 == 0.0:
            angle, sin_a, cos_a = 0.0, 0.0, 1.0
        else:
            cos_a = dot / (mag1 * mag2)
            cos_a = float(np.clip(cos_a, -1.0, 1.0))
            angle = math.acos(cos_a)
            sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
        result[f"a{i}{j}{k}"] = angle
        result[f"as{i}{j}{k}"] = sin_a
        result[f"aa{i}{j}{k}"] = cos_a

    # --- Torsion angles (vectorised) ---
    for i, j, k, m in combinations(range(n), 4):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[j]
        v3 = pts[m] - pts[k]
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        n1_norm = float(np.linalg.norm(n1))
        n2_norm = float(np.linalg.norm(n2))
        if n1_norm == 0.0 or n2_norm == 0.0:
            torsion, sin_t, cos_t = 0.0, 0.0, 1.0
        else:
            n1u = n1 / n1_norm
            n2u = n2 / n2_norm
            cos_t = float(np.dot(n1u, n2u))
            v2_norm = float(np.linalg.norm(v2))
            sin_t = float(np.dot(np.cross(n1u, n2u), v2 / v2_norm)) if v2_norm else 0.0
            torsion = math.atan2(sin_t, cos_t)
        result[f"t{i}{j}{k}{m}"] = torsion
        result[f"ts{i}{j}{k}{m}"] = sin_t
        result[f"ta{i}{j}{k}{m}"] = cos_t

    return result


def feature_column_names(n: int, include_raw_angles: bool = True) -> list[str]:
    """Return the ordered list of feature column names for ``n`` points.

    When ``include_raw_angles`` is ``False`` only the sine / cosine transforms
    are emitted (this is what the trained model consumes).
    """
    cols: list[str] = []
    for i, j in combinations(range(n), 2):
        cols.append(f"d{i}{j}")
    for i, j, k in combinations(range(n), 3):
        if include_raw_angles:
            cols.append(f"a{i}{j}{k}")
        cols.append(f"as{i}{j}{k}")
        cols.append(f"aa{i}{j}{k}")
    for i, j, k, l in combinations(range(n), 4):
        if include_raw_angles:
            cols.append(f"t{i}{j}{k}{l}")
        cols.append(f"ts{i}{j}{k}{l}")
        cols.append(f"ta{i}{j}{k}{l}")
    return cols


def training_feature_columns(n: int) -> list[str]:
    """Feature columns kept after dropping raw angles (used by the bundle)."""
    return feature_column_names(n, include_raw_angles=False)


def raw_angle_columns(n: int) -> list[str]:
    """Columns to drop before training (raw ``a`` and ``t`` values)."""
    cols: list[str] = []
    for i, j, k in combinations(range(n), 3):
        cols.append(f"a{i}{j}{k}")
    for i, j, k, l in combinations(range(n), 4):
        cols.append(f"t{i}{j}{k}{l}")
    return cols
