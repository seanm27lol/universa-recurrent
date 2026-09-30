#!/usr/bin/env bash
# Consistent-edit replication D3 (protocols/edit_replication.md): CPU suite,
# then one run on the Gemma-3-12B calibration split. Requires --lock,
# --calibration-run, --pilot-run and --d1-run (all read-only; never copied).
# Loads the target, AV and AR in turn. Runs exactly once. Mirrors
# run_edit_diagnostics.sh; no hidden setup.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.edit_replication "$@"
