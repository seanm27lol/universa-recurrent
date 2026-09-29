#!/usr/bin/env bash
# Post-Phase-Two steering assay: CPU suite, then one steering run over the
# reused 128-group pilot split. Requires --pilot-run PATH pointing at the
# completed pinned pilot run directory (accessed read-only at run time for
# arm-1 saved AR directions and cross-run checks; never copied). The assay
# loads only the target and AR weights — never the AV. It runs exactly once
# per protocols/steering_assay_brief.md and never reopens the closed pilot.
# Mirrors run_pilot.sh; no hidden setup.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.steering "$@"
