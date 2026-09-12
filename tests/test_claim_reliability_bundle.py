"""The overnight launcher must execute the exact source and preserve arguments."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
bundle = importlib.import_module('build_claim_reliability_bundle')


def source_payload():
    files = {'SOURCE_COMMIT.txt': b'abc123\n',
             'scripts/run_portable_claim_reliability.sh':
                 b'#!/usr/bin/env bash\nset -eu\nprintf "%s\\n%s\\n" "$1" "$2" > "$1/launched.txt"\n'}
    files['SOURCE_MANIFEST.json'] = json.dumps(
        {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}).encode()
    return files


def prepare(tmp_path, archive=None):
    archive = bundle.zip_source(source_payload()) if archive is None else archive
    script = tmp_path / 'study launcher.sh'
    script.write_bytes(bundle.launcher(archive, 'abc123'))
    environment = {**os.environ, 'UNIVERSA_PYTHON': sys.executable,
                   'UNIVERSA_DOWNLOADS': str(tmp_path / 'downloads')}
    return script, environment


def run(script, environment, *args):
    return subprocess.run(['bash', str(script), *map(str, args)], env=environment,
                          text=True, capture_output=True, timeout=20)


def test_source_archive_is_reproducible_and_extraction_does_not_launch(tmp_path):
    files = source_payload()
    archive = bundle.zip_source(files)
    assert archive == bundle.zip_source(files)
    script, environment = prepare(tmp_path, archive)
    result = run(script, environment, '--extract-only')
    assert result.returncode == 0, result.stderr
    roots = list((tmp_path / 'downloads').glob('universa-claims-*'))
    assert len(roots) == 1
    for name, data in files.items():
        assert (roots[0] / bundle.PREFIX / name).read_bytes() == data
    assert not (roots[0] / 'run.pid').exists()


def test_modified_embedded_archive_is_rejected(tmp_path):
    script, environment = prepare(tmp_path)
    data = script.read_bytes()
    marker = b'\n__UNIVERSA_CLAIM_ARCHIVE_BELOW__\n'
    first, payload = data.split(marker)
    payload = (b'A' if payload[:1] != b'A' else b'B') + payload[1:]
    script.write_bytes(first + marker + payload)
    result = run(script, environment, '--extract-only')
    assert result.returncode != 0
    assert 'checksum mismatch' in result.stderr


def test_invalid_manifest_is_rejected(tmp_path):
    files = source_payload()
    files['SOURCE_COMMIT.txt'] = b'changed\n'
    script, environment = prepare(tmp_path, bundle.zip_source(files))
    result = run(script, environment, '--extract-only')
    assert result.returncode != 0
    assert 'manifest mismatch' in result.stderr


def test_archive_traversal_is_rejected_even_with_matching_checksum(tmp_path):
    contents = io.BytesIO()
    with zipfile.ZipFile(contents, 'w') as output:
        output.writestr('../../escaped', 'bad')
    script, environment = prepare(tmp_path, contents.getvalue())
    result = run(script, environment, '--extract-only')
    assert result.returncode != 0
    assert 'Invalid archive path' in result.stderr
    assert not (tmp_path / 'escaped').exists()


@pytest.mark.parametrize('mode', ['foreground', 'background'])
def test_launcher_preserves_paths_and_detaches_when_requested(tmp_path, mode):
    script, environment = prepare(tmp_path)
    replication = tmp_path / 'replication with spaces'
    replication.mkdir()
    args = (['--foreground'] if mode == 'foreground' else []) + [replication, 'cpu']
    result = run(script, environment, *args)
    assert result.returncode == 0, result.stderr
    marker = replication / 'launched.txt'
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert marker.read_text().splitlines() == [str(replication), 'cpu']
    if mode == 'background':
        assert 'Started in background' in result.stdout
        assert len(list((tmp_path / 'downloads').glob('*/run.pid'))) == 1
