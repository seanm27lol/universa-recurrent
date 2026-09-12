"""Package a committed source snapshot as one offline DGX launcher.

The embedded ZIP is identified by SHA-256. Extraction and experiments use fresh
directories; no Git pull, dependency install, or checkpoint overwrite occurs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'universa-claim-reliability-study'
START_HERE = '''# Universa claim reliability study

This snapshot runs the fixed protocol in experiments/claim_reliability_v1.json.
It reuses the five saved replication checkpoints; no training is performed.

From an extracted source folder on the DGX:

```bash
bash scripts/run_portable_claim_reliability.sh
```

The single-file launcher starts a detached run by default, so closing SSH does
not stop it. Its printed console log contains failures or the final upload ZIP.
Use --foreground to keep output in your terminal, or --extract-only to unpack
without testing or launching an experiment. Optional positional arguments are
the replication results directory and device (cuda or cpu).

Defaults use $HOME/projects/universa-recurrent-git/.venv/bin/python and
$HOME/projects/universa-recurrent-git/runs/replication-20260909-173545-3Lc0us/results.
Set UNIVERSA_PYTHON if the existing environment is elsewhere.

This is a planned quality study. Local smoke tests use different input seeds;
no final-test result is claimed by distributing this file. All model thresholds
are saved before any final-test inputs are generated. Reports include synthetic
data and predictions, never checkpoint weights. Read docs/claim_reliability.md.
'''


def source_files(ref: str) -> tuple[str, dict[str, bytes]]:
    commit = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', '--verify', f'{ref}^{{commit}}'], text=True).strip()
    archive = subprocess.check_output(['git', '-C', str(ROOT), 'archive', '--format=tar', commit])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isdir():
                continue
            if not item.isfile() or Path(item.name).is_absolute() or '..' in Path(item.name).parts:
                raise ValueError('Source snapshot contains an unsupported entry')
            files[item.name] = source.extractfile(item).read()
    required = ('experiments/claim_reliability_v1.json', 'scripts/run_claim_reliability.py',
                'scripts/run_portable_claim_reliability.sh')
    if any(name not in files for name in required):
        raise ValueError('Commit does not contain the complete claim-reliability study')
    files['SOURCE_COMMIT.txt'] = (commit + '\n').encode()
    files['START_HERE.md'] = START_HERE.encode()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files['SOURCE_MANIFEST.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    return commit, files


def zip_source(files: dict[str, bytes]) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f'{PREFIX}/{name}', date_time=(2026, 9, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return result.getvalue()


def launcher(archive: bytes, commit: str) -> bytes:
    checksum = hashlib.sha256(archive).hexdigest()
    header = r'''#!/usr/bin/env bash
# Universa claim reliability study. Complete committed source is embedded below.
set -euo pipefail
UNIVERSA_LAUNCH_MODE=background
if [[ "${1:-}" == --foreground || "${1:-}" == --extract-only ]]; then
  UNIVERSA_LAUNCH_MODE="${1#--}"
  shift
fi
if [[ $# -gt 2 || "${1:-}" == --* ]]; then
  echo 'Usage: bash run-universa-claim-reliability.sh [--foreground|--extract-only] [replication-directory] [cuda|cpu]' >&2
  exit 2
fi
export UNIVERSA_PYTHON="${UNIVERSA_PYTHON:-$HOME/projects/universa-recurrent-git/.venv/bin/python}"
if [[ ! -x "$UNIVERSA_PYTHON" ]]; then
  echo "Missing experiment Python: $UNIVERSA_PYTHON" >&2
  echo 'Set UNIVERSA_PYTHON to the existing experiment environment.' >&2
  exit 2
fi
UNIVERSA_REPLICATION="${1:-$HOME/projects/universa-recurrent-git/runs/replication-20260909-173545-3Lc0us/results}"
UNIVERSA_DEVICE="${2:-cuda}"
if [[ "$UNIVERSA_DEVICE" != cuda && "$UNIVERSA_DEVICE" != cpu ]]; then
  echo 'Device must be cuda or cpu.' >&2
  exit 2
fi
if [[ "$UNIVERSA_LAUNCH_MODE" != extract-only && ! -d "$UNIVERSA_REPLICATION" ]]; then
  echo "Missing checkpoint directory: $UNIVERSA_REPLICATION" >&2
  exit 2
fi
UNIVERSA_DOWNLOADS="${UNIVERSA_DOWNLOADS:-$HOME/Downloads}"
mkdir -p "$UNIVERSA_DOWNLOADS"
UNIVERSA_STUDY_DIR="$(mktemp -d "$UNIVERSA_DOWNLOADS/universa-claims-XXXXXX")"
"$UNIVERSA_PYTHON" - "$0" "$UNIVERSA_STUDY_DIR" <<'UNIVERSA_EXTRACT_PY'
import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
script = Path(sys.argv[1]).read_bytes()
payload = script.split(b'\n__UNIVERSA_CLAIM_ARCHIVE_BELOW__\n', 1)[1]
archive = base64.b64decode(b''.join(payload.split()), validate=True)
if hashlib.sha256(archive).hexdigest() != '__ARCHIVE_SHA__':
    raise SystemExit('Embedded source archive checksum mismatch.')
root = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    names = bundle.namelist()
    if len(names) != len(set(names)) or bundle.testzip() is not None:
        raise SystemExit('Invalid source archive.')
    for name in names:
        if not (root / name).resolve().is_relative_to(root):
            raise SystemExit('Invalid archive path.')
    bundle.extractall(root)
source = root / 'universa-claim-reliability-study'
manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
for name, expected in manifest.items():
    path = source / name
    if not path.resolve().is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Source manifest mismatch: ' + name)
print(f'Source unpacked to {source}', flush=True)
UNIVERSA_EXTRACT_PY
echo 'Source commit: __SOURCE_COMMIT__'
if [[ "$UNIVERSA_LAUNCH_MODE" == extract-only ]]; then
  exit 0
fi
UNIVERSA_RUNNER="$UNIVERSA_STUDY_DIR/universa-claim-reliability-study/scripts/run_portable_claim_reliability.sh"
if [[ "$UNIVERSA_LAUNCH_MODE" == foreground ]]; then
  bash "$UNIVERSA_RUNNER" "$UNIVERSA_REPLICATION" "$UNIVERSA_DEVICE"
else
  UNIVERSA_LOG="$UNIVERSA_STUDY_DIR/console.log"
  nohup bash "$UNIVERSA_RUNNER" "$UNIVERSA_REPLICATION" "$UNIVERSA_DEVICE" > "$UNIVERSA_LOG" 2>&1 < /dev/null &
  UNIVERSA_PID=$!
  echo "$UNIVERSA_PID" > "$UNIVERSA_STUDY_DIR/run.pid"
  echo "Started in background (PID $UNIVERSA_PID). Closing SSH will not stop it."
  echo "Log: $UNIVERSA_LOG"
  printf 'To view progress: tail -f %q\n' "$UNIVERSA_LOG"
  echo 'On success, upload claim-reliability-*-reports.zip from Downloads.'
  echo 'A launched process is not a completed result; check the log if no ZIP appears.'
fi
exit 0
__UNIVERSA_CLAIM_ARCHIVE_BELOW__
'''
    header = header.replace('__ARCHIVE_SHA__', checksum).replace('__SOURCE_COMMIT__', commit)
    encoded = '\n'.join(textwrap.wrap(base64.b64encode(archive).decode('ascii'), 76)) + '\n'
    return header.encode() + encoded.encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD', help='Committed source revision; uncommitted changes are excluded')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    commit, files = source_files(args.ref)
    archive = zip_source(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zipped = args.output_dir / f'{PREFIX}.zip'
    script = args.output_dir / 'run-universa-claim-reliability.sh'
    for path, content in ((zipped, archive), (script, launcher(archive, commit))):
        with path.open('xb') as handle:
            handle.write(content)
    script.chmod(0o755)
    print(json.dumps({'commit': commit, 'archive': str(zipped), 'launcher': str(script),
                     'archive_sha256': hashlib.sha256(archive).hexdigest(),
                     'launcher_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
