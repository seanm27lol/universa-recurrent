"""Check the packing list before a run: one version, one install, every module."""
from __future__ import annotations
import ast
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('cli.py', 'neural/model.py', 'neural/train.py', 'neural/lingua.py',
            'neural/verification.py', 'neural/v2.py', 'neural/v2_train.py',
            'neural/v2_lingua.py', 'neural/v2_verification.py', 'neural/dual_output.py',
            'neural/dual_study.py', 'neural/dual_lingua.py', 'neural/dual_cli.py')


def main():
    import universa_recurrent as package
    installed = Path(package.__file__).resolve()
    expected = ROOT/'src/universa_recurrent/__init__.py'
    if installed != expected:
        raise SystemExit(f'Wrong editable install: {installed}; reinstall this checkout')
    config = tomllib.loads((ROOT/'pyproject.toml').read_text('utf-8'))
    if ('version' in config['project'] or
        config['tool']['setuptools']['dynamic']['version']['attr'] != 'universa_recurrent.__version__'):
        raise SystemExit('Version must have one authoritative source')
    if version('universa-recurrent') != package.__version__:
        raise SystemExit('Installed metadata disagrees; reinstall after pulling a version change')
    for relative in REQUIRED:
        path = installed.parent/relative
        if not path.is_file():
            raise SystemExit(f'Missing release module: {relative}')
        ast.parse(path.read_text('utf-8'), filename=str(path))
    help_text = subprocess.check_output([sys.executable,'-m','universa_recurrent.cli','--help'],text=True)
    for command in ('neural-train','neural-v2-train','neural-v2-eval','neural-v2-verify'):
        if command not in help_text:
            raise SystemExit(f'Missing CLI command: {command}')
    if find_spec('torch') is not None:
        help_text = subprocess.check_output([sys.executable,'-m','universa_recurrent.neural.dual_cli','--help'],text=True)
        for command in ('compare','replicate','verify'):
            if command not in help_text:
                raise SystemExit(f'Missing dual command: {command}')
    print(f'Release check PASS: {package.__version__}')
    print('Version, editable-install location, required modules, and available CLI commands agree.')
    print('Packaging validation is not a model-quality or GPU-performance result.')


if __name__ == '__main__':
    main()
