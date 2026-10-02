#!/usr/bin/env bash
# Oracle-text control D4 (protocols/edit_oracle_control.md): CPU suite, then
# one run on D3's primary receivers. Requires --lock, --d3-run and
# --calibration-run (read-only; never copied). Loads the AR and target only.
# Runs exactly once. Mirrors run_edit_replication.sh; no hidden setup.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.edit_oracle_control "$@"
