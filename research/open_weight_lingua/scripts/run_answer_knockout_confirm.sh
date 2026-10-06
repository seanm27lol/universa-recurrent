#!/usr/bin/env bash
# Phase Eight confirmatory knockout on validation_b
# (protocols/phase_eight_confirmatory_validation_b.md): CPU suite, then one
# family's run. Requires --lock; pass --cache when the locked target lives
# elsewhere. Loads the target only. Evaluate afterwards on CPU with
#   python -m open_weight_lingua.answer_knockout_confirm evaluate --runs GEMMA_RUN QWEN_RUN --output SUMMARY.json
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.answer_knockout_confirm run "$@"
