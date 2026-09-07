"""Run after installing the package; no GPU or network access required."""
from universa_recurrent.cli import main
if __name__ == "__main__":
    raise SystemExit(main(["demo", "--trace", "compact"]))
