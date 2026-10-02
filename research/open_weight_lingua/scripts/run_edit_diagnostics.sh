#!/usr/bin/env bash
# Edit-channel diagnostics (protocols/edit_channel_diagnostics.md): CPU suite,
# then one diagnostic part. Requires --part d1|d2, --lock and --pilot-run PATH
# pointing at the matching completed confirmation pilot (read-only; never
# copied). D1 loads the AR and target only; D2 (Gemma-3-12B) loads the target
# and AV. Each part runs exactly once. Mirrors run_steering.sh; no hidden setup.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"
phase2_setup_start=$SECONDS
command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }
UV_PROJECT_ENVIRONMENT="${phase2_repo}/.venv-phase2" uv sync --project "${phase2_project}" --extra test --locked
"${phase2_repo}/.venv-phase2/bin/python" -m pytest -q -c "${phase2_project}/pyproject.toml" "${phase2_project}/tests"
export OWL_SETUP_SECONDS=$((SECONDS - phase2_setup_start))
exec "${phase2_repo}/.venv-phase2/bin/python" -m open_weight_lingua.edit_diagnostics "$@"
