#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topological (chemistry-anchored) ordering of a double-tetrad's 8 guanines.

The geometric canonization in ``canonical_order.py`` is deterministic per point
set but **not consistent across structures**: its rotation-start tiebreak and
PCA axis signs are anchored to the crystal frame, so two topologically
identical double tetrads can be written in different cyclic phases and the
indexed features (``d01`` != ``d02``) end up meaning different physical pairs
in different files.

This module replaces the geometric canonization for *positives* with an order
derived from the ElTetrado annotation itself:

1. **Columns.**  In ``double_tetrad_motifs_by_pdb.json`` the ``unit_ids`` of a
   motif are ``[lo_nt1..lo_nt4, hi_nt1..hi_nt4]`` where ``nt1..nt4`` is the
   cyclic walk around a tetrad (this is the walk the ONZ classification is
   defined on — verified empirically: consecutive walk neighbours are at
   in-plane side distance, ~0.7x the diagonal, as expected for a square).
   Column ``i`` is therefore the pair ``(unit_ids[i], unit_ids[4+i])``.
2. **Cyclic direction.**  The stacking normal is
   ``centroid(upper) - centroid(lower)``; the signed area of the lower tetrad's
   C1' polygon in ElTetrado walk order decides whether the walk goes
   counter-clockwise or clockwise around that normal.  This uses only relative
   geometry, so it is invariant to rigid motion of the input structure —
   unlike the PCA-based handedness fix, which is frame-dependent.
3. **Start column.**  Column 1 is the column whose *lower* nucleotide has the
   smallest ``(chain_id, residue_number, insertion_code)`` — a deterministic,
   reproducible tiebreak on chemistry identifiers.

The resulting order is column-major interleaved::

    [lo_1, hi_1, lo_2, hi_2, lo_3, hi_3, lo_4, hi_4]

so e.g. ``d01`` is the intra-column stacking distance (~3.7 A) and ``d02`` the
in-plane C1'-C1' distance within the lower tetrad.

The same file also provides helpers used by the augmentation in
``04-generate-csv.py``: ``column_sequence_from_perm`` labels an arbitrary
ordering with column indices, which lets us recognise which phase variants the
geometric canonizer (still used for negatives and inference candidates) can
emit for a given motif.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float, float]

# In the column-major interleaved topological order, the residue at position p
# belongs to column p // 2 (positions 0,1 -> column 0; 2,3 -> column 1; ...).
COLUMN_OF_POSITION = tuple(p // 2 for p in range(8))


def _residue_sort_key(unit: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(unit.get("chain_id") or ""),
        int(unit.get("residue_number") or 0),
        str(unit.get("insertion_code") or ""),
    )


def topological_order_indices(
    unit_ids: List[Dict[str, Any]], coords: Sequence[Point]
) -> List[int]:
    """Return the permutation of the 8 unit-ordered residues that yields the
    topological order ``[lo_1, hi_1, lo_2, hi_2, lo_3, hi_3, lo_4, hi_4]``.

    Parameters
    ----------
    unit_ids
        The motif's ``unit_ids`` in ElTetrado order: the first four are the
        lower tetrad's ``nt1..nt4``, the last four the upper tetrad's.
    coords
        The C1' coordinates of those 8 residues, in the same order.
    """
    if len(unit_ids) != 8 or len(coords) != 8:
        raise ValueError(f"expected 8 unit_ids / coords, got {len(unit_ids)} / {len(coords)}")

    pts = np.asarray(coords, dtype=float)
    lower_centroid = pts[:4].mean(axis=0)
    upper_centroid = pts[4:].mean(axis=0)
    normal = upper_centroid - lower_centroid
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        raise ValueError("degenerate double tetrad: upper and lower centroids coincide")
    normal = normal / norm

    # Signed area of the lower tetrad's C1' polygon in ElTetrado walk order,
    # measured around the stacking normal.  Positive => the walk 1->2->3->4 is
    # counter-clockwise when viewed along +normal (from lower towards upper).
    centred = pts[:4] - lower_centroid
    signed_area = 0.5 * float(
        np.sum(np.cross(centred, np.roll(centred, -1, axis=0)) @ normal)
    )
    walk = [0, 1, 2, 3] if signed_area > 0 else [0, 3, 2, 1]

    # Start at the column whose lower nucleotide has the smallest identifier.
    start = min(walk, key=lambda col: _residue_sort_key(unit_ids[col]))
    start_pos = walk.index(start)
    walk = walk[start_pos:] + walk[:start_pos]

    perm: List[int] = []
    for col in walk:
        perm.append(col)       # lower tetrad, this column
        perm.append(col + 4)   # upper tetrad, same column
    return perm


def column_sequence_from_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    """Map a permutation of the topological order to its column sequence.

    ``perm[p]`` is the index (into the topologically ordered residues) of the
    residue that ends up at position ``p``; the result labels each position
    with its topological column (0-3).  Used to deduplicate phase variants and
    to reject shuffled orderings that coincide with valid phases.
    """
    return tuple(COLUMN_OF_POSITION[i] for i in perm)


def tetrad_walk_stats(coords: Sequence[Point]) -> Tuple[float, float]:
    """Sanity-check the ElTetrado cyclic walk on one tetrad's 4 C1' points.

    Returns ``(mean_consecutive, mean_diagonal)`` in-plane distances.  For a
    genuine perimeter walk the ratio is ~0.7 (side over diagonal of a square);
    a criss-cross ordering would push the ratio towards or above 1.
    """
    pts = np.asarray(coords, dtype=float)
    centred = pts - pts.mean(axis=0)
    d = np.linalg.norm(centred[:, None, :] - centred[None, :, :], axis=-1)
    consecutive = [float(d[i, (i + 1) % 4]) for i in range(4)]
    diagonal = [float(d[0, 2]), float(d[1, 3])]
    return float(np.mean(consecutive)), float(np.mean(diagonal))
