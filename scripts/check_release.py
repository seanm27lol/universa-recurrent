"""Catch incomplete releases and stale editable installs before a training run.

Run from a Git checkout: python scripts/check_release.py
Like checking a packing list before a trip, this verifies that required modules,
commands, and the installed package identity belong to the same release.
"""
from __future__ import annotations
import ast
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "cli.py", "neural/model.py", "neural/train.py", "neural/lingua.py",
    "neural/verification.py", "neural/v2.py", "neural/v2_train.py",
    "neural/v2_lingua.py", "neural/v2_verification.py",
)


def main() -> None:
    import universa_recurrent as package
    installed = Path(package.__file__).resolve()
    expected = ROOT / "src/universa_recurrent/__init__.py"
    if installed != expected:
        raise SystemExit(f"Wrong editable install: {installed}\nRun: python -m pip install -e '.[test,neural]'")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in config["project"], "Version must have one source"
    assert config["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "universa_recurrent.__version__"
    assert version("universa-recurrent") == package.__version__, "Reinstall after pulling a version change"
    for relative in REQUIRED:
        path = installed.parent / relative
        if not path.is_file():
            raise SystemExit(f"Missing release module: {relative}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    help_text = subprocess.check_output(
        [sys.executable, "-m", "universa_recurrent.cli", "--help"], text=True)
    for command in ("neural-train", "neural-eval", "neural-verify", "neural-v2-train",
                    "neural-v2-eval", "neural-v2-benchmark", "neural-v2-demo", "neural-v2-verify"):
        assert command in help_text, f"Missing CLI command: {command}"
    print(f"Release check PASS: {package.__version__}")
    print("Version, editable-install location, required modules, and CLI commands agree.")
    print("This is a packaging check, not a model-quality or GPU-performance claim.")


if __name__ == "__main__":
    main()
