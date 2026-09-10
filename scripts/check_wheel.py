"""Build and import away from src/, catching omitted wheel modules and stale versions."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import universa_recurrent
    with tempfile.TemporaryDirectory(prefix='universa-wheel-') as directory:
        tmp=Path(directory)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
                        '--wheel-dir',str(tmp),str(ROOT)],check=True)
        wheel,=tmp.glob('*.whl')
        with zipfile.ZipFile(wheel) as archive:
            for module in ('model','train','lingua','verification','v2','v2_train','v2_lingua',
                           'v2_verification','dual_output','dual_study','dual_lingua','dual_cli','final_state','retention_lingua','retention_study'):
                if f'universa_recurrent/neural/{module}.py' not in archive.namelist():
                    raise RuntimeError(f'Wheel missing module: {module}')
        target=tmp/'installed'
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheel)],check=True)
        env=dict(os.environ,PYTHONPATH=str(target))
        code=('from pathlib import Path; from importlib.metadata import version; '
              'import universa_recurrent as u; '
              f'assert Path(u.__file__).is_relative_to(Path({str(target)!r})); '
              f'assert u.__version__ == version("universa-recurrent") == {universa_recurrent.__version__!r}; '
              'from universa_recurrent.cli import main; main(["--help"])')
        subprocess.run([sys.executable,'-c',code],cwd=tmp,env=env,check=True)
        from importlib.util import find_spec
        if find_spec('torch') is not None:
            subprocess.run([sys.executable,'-m','universa_recurrent.neural.dual_cli','--help'],
                           cwd=tmp,env=env,check=True)
            subprocess.run([sys.executable,'-m','universa_recurrent.neural.retention_study','--help'],
                           cwd=tmp,env=env,check=True)
    print('Wheel check PASS: neural runtimes and dual-output experiment are included.')


if __name__ == '__main__':
    main()
