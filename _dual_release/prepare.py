"""One-off server-side transfer helper; never included in the released source."""
import base64
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import subprocess
import sys

BASE='ae6201e78fc3c4a02e6445e8b44a1979e2a61e91'
SHA='1dffd2837579e0f4a06cbd287c614f2ffccf4ece050ff74eb351ff79ed8a3611'

def git(*args):
    return subprocess.check_output(['git',*args],text=True).strip()

stage=Path(sys.argv[1]).resolve()
root=Path.cwd().resolve()
if git('rev-parse','HEAD') != BASE or git('status','--porcelain'):
    raise SystemExit('Refusing a different base or dirty build checkout')
encoded=''.join((stage/f'part{i:02}.b64').read_text('ascii').strip() for i in range(1,5))
packed=base64.b64decode(encoded,validate=True)
if hashlib.sha256(packed).hexdigest()!=SHA:
    raise SystemExit('Source transfer checksum mismatch')
data=json.loads(lzma.decompress(packed))
if data['base']!=BASE or data['version']!='0.5.0' or len(data['files'])!=13:
    raise SystemExit('Unexpected manifest')
for name,entry in data['files'].items():
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or '\0' in name:
        raise SystemExit('Unsafe path')
    if not (name=='README.md' or name.startswith(('src/universa_recurrent/','tests/','scripts/','docs/','experiments/'))):
        raise SystemExit('Path outside source allowlist')
    target=root/name
    if not target.resolve().is_relative_to(root) or target.is_symlink():
        raise SystemExit('Symlink escape')
    raw=entry['content'].encode('utf-8')
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
        raise SystemExit('Per-file hash mismatch: '+name)
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['git_blob']:
        raise SystemExit('Git blob mismatch: '+name)
for name,entry in data['files'].items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(entry['content'].encode());p.chmod(0o644)
subprocess.run(['git','add','--',*sorted(data['files'])],check=True)
if set(git('diff','--cached','--name-only').splitlines()) != set(data['files']):
    raise SystemExit('Unexpected changed-file set')
tree=git('write-tree')
for name,entry in data['files'].items():
    if git('rev-parse',tree+':'+name)!=entry['git_blob']:
        raise SystemExit('Index bytes differ: '+name)
(stage/'verified-tree').write_text(tree+'\n')
print('Verified all 13 source/test/doc files. Complete tree:',tree)
