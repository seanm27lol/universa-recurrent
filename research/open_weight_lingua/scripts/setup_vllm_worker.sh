#!/usr/bin/env bash
# Create the isolated vLLM worker environment for the opt-in AV/AR backend.
#
# vllm==0.30.0 pins torch 2.13.x and transformers 5.x, which hard-conflict with
# the pipeline's pinned transformers 4.57.6, so the worker can never be
# co-installed in .venv-phase2. This script creates a SEPARATE venv at
# research/open_weight_lingua/.venv-vllm (gitignored) and never touches
# .venv-phase2 or any other environment.
set -euo pipefail
phase2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${phase2_project}/.venv-vllm"
pin="vllm==0.30.0"
lock_listing="${venv}/vllm-worker-lock.txt"

command -v uv >/dev/null || { echo 'Install uv first; see the Phase Two README.' >&2; exit 2; }

if [[ -e "${venv}" ]]; then
  if [[ -x "${venv}/bin/python" ]] && \
     "$("${venv}/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)" == "0.30.0" ]]; then
    echo "Reusing existing ${venv} (vllm 0.30.0 present)."
  else
    echo "Refusing to reuse ${venv}: missing, or wrong vllm version." >&2
    echo "Remove the directory and re-run to rebuild it." >&2
    exit 1
  fi
else
  # Python 3.11 matches the pipeline interpreter family and the measured spike.
  uv venv --python 3.11 "${venv}"
  uv pip install --python "${venv}/bin/python" "${pin}"
fi

"${venv}/bin/python" - <<'PY'
import vllm, torch, transformers, safetensors
assert vllm.__version__ == "0.30.0", vllm.__version__
print(f"vllm {vllm.__version__}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, safetensors {safetensors.__version__}")
PY

# Record the exact resolved environment next to the venv (gitignored with it).
{
  echo "# vllm worker lock listing, created $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# pin: ${pin}; python: $("${venv}/bin/python" -VV 2>&1)"
  uv pip freeze --python "${venv}/bin/python"
} > "${lock_listing}"
echo "Recorded lock listing: ${lock_listing}"
echo "sha256: $(sha256sum "${lock_listing}" | cut -d' ' -f1)"
echo "Note: .venv-phase2 was not read or modified."
