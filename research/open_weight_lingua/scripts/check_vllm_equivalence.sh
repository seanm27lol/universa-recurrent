#!/usr/bin/env bash
# Measured vLLM/eager equivalence gate for the opt-in AV/AR backend.
#
# Replays a completed eager run's saved activations and descriptions through the
# vLLM worker and compares against the saved eager outputs (AV token identity,
# AR direction cosine). The reference run is read-only. Exit 0 on PASS, 1 on
# FAIL, 2 on setup errors. A FAIL or partial result keeps the vllm backend
# non-default and forbids mixing its outputs into eager-regime evidence.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
phase2_repo="$(cd -- "${phase2_project}/../.." && pwd)"

# The pinned eager smoke bundle (read-only reference) from the milestone runs.
reference_default="/home/seanjazm27/projects/universa-recurrent-m2/research/open_weight_lingua/runs/smoke-20260921T233021Z-cbe4057e"
run_dir="${1:-${reference_default}}"
output="${2:-${phase2_project}/runs/gate-$(date -u +%Y%m%dT%H%M%SZ)}"

worker="${OWL_VLLM_PYTHON:-${phase2_project}/.venv-vllm/bin/python}"
if [[ ! -x "${worker}" ]]; then
  echo "vLLM worker missing: ${worker}" >&2
  echo "Run scripts/setup_vllm_worker.sh first (or set OWL_VLLM_PYTHON)." >&2
  exit 2
fi
# The gate module runs under the pipeline interpreter (transformers 4.57.6);
# OWL_PHASE2_PYTHON overrides the default in-tree .venv-phase2 location.
phase2_python="${OWL_PHASE2_PYTHON:-${phase2_repo}/.venv-phase2/bin/python}"
if [[ ! -x "${phase2_python}" ]]; then
  echo "Pipeline python missing: ${phase2_python}" >&2
  echo "Run scripts/run_smoke.sh once to create .venv-phase2, or set OWL_PHASE2_PYTHON." >&2
  exit 2
fi
if [[ ! -d "${run_dir}" ]]; then
  echo "Reference run missing: ${run_dir}" >&2
  exit 2
fi

cd "${phase2_project}"
PYTHONPATH="${phase2_project}/src" OWL_VLLM_PYTHON="${worker}" \
  "${phase2_python}" -m open_weight_lingua.vllm_equivalence \
  --run-dir "${run_dir}" --output "${output}" "${@:3}"
