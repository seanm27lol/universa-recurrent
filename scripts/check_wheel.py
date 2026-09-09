"""Build and import the wheel away from src/, so missing packaged files cannot hide."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import universa_recurrent
    with tempfile.TemporaryDirectory(prefix="universa-wheel-") as directory:
        tmp = Path(directory)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(tmp), str(ROOT)], check=True)
        wheel, = tmp.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            for module in ("model", "train", "lingua", "verification", "v2", "v2_train",
                           "v2_lingua", "v2_verification"):
                assert f"universa_recurrent/neural/{module}.py" in archive.namelist(), module
        target = tmp / "installed"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
                        str(target), str(wheel)], check=True)
        environment = dict(os.environ, PYTHONPATH=str(target))
        check = (
            "from pathlib import Path; from importlib.metadata import version; "
            "import universa_recurrent as u; "
            f"assert Path(u.__file__).is_relative_to(Path({str(target)!r})); "
            f"assert u.__version__ == version('universa-recurrent') == {universa_recurrent.__version__!r}; "
            "from universa_recurrent.cli import main; main(['--help'])"
        )
        subprocess.run([sys.executable, "-c", check], cwd=tmp, env=environment, check=True)
    print("Wheel check PASS: independently installed package has both neural generations.")


if __name__ == "__main__":
    main()
