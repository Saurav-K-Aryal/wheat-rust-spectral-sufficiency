#!/usr/bin/env bash
# Rebuild the numbers, tables, and figures of the paper.
#
#   bash reproduce.sh          from the shipped results (about two minutes, no image data needed)
#   bash reproduce.sh --all    from the images first (roughly 85 CPU-hours; set WORKERS to use more cores)
#
# Set PYTHON to choose the interpreter, WHEATRUST_DATA if the images are not in ./data, and WHEATRUST_MODELS if the
# TabPFN checkpoint is not in ./models.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
W=${WORKERS:-1}

if [[ "${1:-}" == "--all" ]]; then
  $PY scripts/eda/01_audit.py
  $PY scripts/eda/02_overlap_offsets.py
  $PY scripts/training/01_make_folds.py
  $PY scripts/training/02_build_features.py
  $PY scripts/training/03_stretch_control.py
  $PY scripts/eda/03_adversarial_validation.py
  $PY scripts/eda/04_band_separability.py
  $PY scripts/training/04_band_selection.py --workers "$W"
  $PY scripts/training/05_run_units.py --exp E1 E2 E3f E3t E4 E4t E5 E9 E9t --workers "$W"
  $PY scripts/training/06_window_permutation.py --workers "$W"
  $PY scripts/training/07_crossbatch.py --workers "$W"
  $PY scripts/training/08_submissions.py
fi

# the analyses run in number order (01 to 14), then the tables and figures
for s in scripts/analysis/[0-9]*.py scripts/tables/[0-9]*.py scripts/plots/[0-9]*.py; do
  echo "== $s"
  $PY "$s"
done
echo "Done. The paper's numbers, tables, and figures are in outputs/, the robustness analyses in outputs/robustness/."
