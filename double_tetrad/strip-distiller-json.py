#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strip distiller output JSONs to the subset needed for training.

``distiller`` writes rich JSON files containing, besides the clustering
itself, full diagnostics (selection curve, PCA axes, fit data, linkage
matrices).  For the 2541-structure tetrad dataset these diagnostics push
``approximate-hierarchical.json`` to ~620 MB, which is far too large to
commit.

The training notebook only ever reads ``data["clustering"]["clusters"]``, so
this script loads each ``approximate-*.json`` produced by ``distiller`` and
re-writes it in-place keeping only:

    {
      "clustering": {
        "clusters": [ ... ]
      }
    }

This brings each file down to a few tens of kilobytes, matching the size of
the equivalent files in the GNRA pipeline (project root).

Usage
-----
    uv run python tetrad/strip-distiller-json.py
    # or with explicit files
    uv run python tetrad/strip-distiller-json.py tetrad/approximate-hierarchical.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def strip_file(path: str) -> tuple[int, int]:
    """Strip one distiller JSON in place. Returns ``(before_bytes, after_bytes)``."""
    before = os.path.getsize(path)
    with open(path) as f:
        data = json.load(f)

    clustering = data.get("clustering", {})
    clusters = clustering.get("clusters", [])

    slim = {"clustering": {"clusters": clusters}}

    with open(path, "w") as f:
        json.dump(slim, f, indent=2)

    after = os.path.getsize(path)
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="distiller JSON files to strip (default: all tetrad/approximate-*.json)",
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        tetrad_dir = Path(__file__).parent
        files = sorted(str(p) for p in tetrad_dir.glob("approximate-*.json"))

    if not files:
        print("No files to strip.")
        return

    for fp in files:
        before, after = strip_file(fp)
        print(
            f"  {fp}: {before / 1e6:.1f} MB -> {after / 1e3:.1f} KB "
            f"({before / max(after, 1):.0f}x smaller)"
        )


if __name__ == "__main__":
    main()
