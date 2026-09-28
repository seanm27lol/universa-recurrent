#!/usr/bin/env bash
# Milestone 2 pilot: CPU suite, then the 128-group pilot. Requires
# --calibration-fit PATH pointing at a completed calibration run directory
# (holding baseline_fit.json/.safetensors + calibration_median_norm.json);
# a direct path to baseline_fit.safetensors also works.
# The pilot never starts validation. Mirrors run_smoke.sh; no hidden setup.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.runner --stage pilot "$@"
