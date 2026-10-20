#! /usr/bin/env bash
# Step 07 — run distiller clustering on the topologically-ordered positives.
#
# Runs the three approximate-mode clustering methods used by the training
# notebook (hierarchical, affinity-propagation, facility-location) over
# motif_cif_files/*.cif, then strips the raw distiller JSONs in place with
# strip-distiller-json.py (the raw diagnostics are far too large to keep).
#
# Usage
# -----
#     uv run bash double_tetrad/07-run-clustering.sh [--force]
#
# Without --force, methods whose output JSON already exists are skipped.
set -euo pipefail

# Repo root = parent of this script's directory; uv run needs pyproject.toml.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

force=0
if [[ "${1:-}" == "--force" ]]; then
    force=1
elif [[ -n "${1:-}" ]]; then
    echo "Unknown argument: $1 (usage: $0 [--force])" >&2
    exit 2
fi

cif_dir="double_tetrad/motif_cif_files"
cifs=(${cif_dir}/*.cif)
if [[ ! -e "${cifs[0]}" ]]; then
    echo "No CIF files in ${cif_dir}/ — run 02-generate-positive.py first." >&2
    exit 1
fi
echo "Clustering ${#cifs[@]} CIF files from ${cif_dir}/"

for method in hierarchical affinity-propagation facility-location; do
    output="double_tetrad/approximate-${method}.json"
    if [[ ${force} -eq 0 && -f "${output}" ]]; then
        echo "Skipping ${method} (${output} exists; use --force to redo)"
        continue
    fi
    echo "Running distiller: approximate / ${method} ..."
    uv run distiller \
        --mode approximate \
        --method "${method}" \
        --output-json "${output}" \
        "${cifs[@]}"
done

echo "Stripping distiller JSONs in place ..."
uv run python double_tetrad/strip-distiller-json.py

echo
echo "Resulting cluster JSONs:"
for method in hierarchical affinity-propagation facility-location; do
    output="double_tetrad/approximate-${method}.json"
    n_clusters="$(uv run python -c "
import json, sys
with open('${output}') as f:
    clusters = json.load(f)['clustering']['clusters']
print(len(clusters))
")"
    printf '  %-22s %8s  %s clusters\n' "${method}" "$(du -h "${output}" | cut -f1)" "${n_clusters}"
done
