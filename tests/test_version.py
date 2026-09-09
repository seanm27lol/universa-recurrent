"""One version source, plus checks that a release actually contains its features."""
from importlib.metadata import version
from pathlib import Path
import tomllib

import pytest
import universa_recurrent
from universa_recurrent.cli import main


def test_version_is_single_sourced_and_matches_installed_metadata():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "universa_recurrent.__version__"
    assert version("universa-recurrent") == universa_recurrent.__version__


def test_cli_reports_imported_version(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == universa_recurrent.__version__


def test_release_contains_both_neural_generations():
    package = Path(universa_recurrent.__file__).resolve().parent
    for name in ("model", "train", "lingua", "verification", "baselines", "data",
                 "v2", "v2_train", "v2_lingua", "v2_verification"):
        assert (package / "neural" / f"{name}.py").is_file(), name
