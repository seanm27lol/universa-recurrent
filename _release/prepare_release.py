"""One-time staging helper. Not part of the release branch.

Reconstruct only a reviewed, checksummed source overlay on its exact Git base.
Never consume files from a user's DGX or publish checkpoints/run artifacts.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

BASE = '45e47cb156d1ebbb63d2751058c109dcc9e5a96b'
ARCHIVE_SHA256 = 'cf155699cdadacd7935ae56966c5dbd65b7100444266bb5da17413820baada8d'
SUBTREES = {'src': 'a5b3ddfcb57b922ada98a1321979c39970388aa7',
            'tests': 'faaa157f87a0dd4a4282de65292084a44f292ec0'}
ALLOWED_ROOT_FILES = {'AGENTS.md', 'Makefile', 'README.md', 'pyproject.toml'}


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True).strip()


def main() -> None:
    staging = Path(sys.argv[1]).resolve()
    root = Path.cwd().resolve()
    if git('rev-parse', 'HEAD') != BASE:
        raise SystemExit('Refusing a different base commit')
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise SystemExit('Refusing a dirty build checkout')
    encoded = ''.join((staging / f'source.{i:02d}.b64').read_text('ascii') for i in range(1, 7))
    archive = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit('Source-transfer checksum mismatch')
    payloads: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as tar:
        members = tar.getmembers()
        if len(members) != 31 or sum(m.size for m in members) > 2_000_000:
            raise SystemExit('Unexpected archive layout or size')
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise SystemExit('Non-regular or unsafe source entry')
            if member.name in payloads:
                raise SystemExit('Duplicate source entry')
            handle = tar.extractfile(member)
            if handle is None:
                raise SystemExit('Unreadable source entry')
            payloads[member.name] = handle.read()
    manifest = json.loads(payloads.pop('_MANIFEST.json'))
    if manifest['base_commit'] != BASE or manifest['version'] != '0.4.1':
        raise SystemExit('Wrong release manifest')
    if manifest['subtrees'] != SUBTREES or set(manifest['files']) != set(payloads):
        raise SystemExit('Manifest/file-set mismatch')
    for name, data in payloads.items():
        allowed = (name in ALLOWED_ROOT_FILES or name == '.github/workflows/tests.yml'
                   or name.startswith(('src/universa_recurrent/', 'tests/', 'docs/', 'scripts/', 'experiments/')))
        if not allowed or '\\' in name or '\0' in name or '\n' in name:
            raise SystemExit(f'Unapproved path: {name!r}')
        entry = manifest['files'][name]
        if entry['mode'] != 0o644 or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise SystemExit(f'Per-file checksum/mode mismatch: {name}')
        target = root / name
        if target.resolve().is_relative_to(root) is False or target.is_symlink():
            raise SystemExit('Symlink/path escape refused')
        data.decode('utf-8')
    # All payloads are validated before the first write. Existing unrelated files
    # are retained from the immutable base, not imported from a developer folder.
    for name, data in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
    subprocess.run(['git', 'add', '--', *sorted(payloads)], check=True)
    tree = git('write-tree')
    for directory, expected in SUBTREES.items():
        if git('rev-parse', f'{tree}:{directory}') != expected:
            raise SystemExit(f'{directory} differs from locally tested source')
    (staging / 'expected-release-tree.txt').write_text(tree + '\n', encoding='ascii')
    print(f'Verified {len(payloads)} UTF-8 files; src and tests match audited Git trees.')
    print(f'Candidate complete tree: {tree}')


if __name__ == '__main__':
    main()
