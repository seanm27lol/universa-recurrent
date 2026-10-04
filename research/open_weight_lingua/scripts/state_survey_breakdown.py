"""Post-hoc breakdown of a Stage 0 survey: is the probe reading the asked variable or both?

Usage:
    python scripts/state_survey_breakdown.py RUN_DIR POSITION:LAYER [POSITION:LAYER ...]

For each site, the Stage 0 one-hot probe is refit on the training split with the
alpha the survey chose. Its held-out accuracy is then split by whether the
variable was the one the prompt asked about. This is descriptive and can never
change gate G0. Example: at the answer position a probe can be right about x on
"What is x?" prompts while knowing nothing about x on "What is y?" prompts; the
aggregate accuracy hides that.
"""

import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open

from open_weight_lingua.state_survey import VALUES


def breakdown(run: Path, position: str, layer: int) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    results = {(r["position"], r["layer"]): r for r in json.loads((run / "results.json").read_text())}
    rows = manifest["rows"]
    part = np.array([r["part"] for r in rows])
    asked_variable = np.array([r["id"].rsplit("-", 1)[1] for r in rows])
    with safe_open(str(run / "raw" / "features.safetensors"), framework="numpy") as handle:
        features = handle.get_tensor(position)[:, layer].astype(np.float64)
    train, test = part == "train", part == "test"
    mean = features[train].mean(axis=0, keepdims=True)
    centred_train, centred_test = features[train] - mean, features[test] - mean
    gram = centred_train @ centred_train.T
    scale = float(np.mean(np.diag(gram)))
    out = {"position": position, "layer": layer}
    for variable in ("x", "y"):
        values = np.array([r["labels"][variable]["value"] for r in rows])
        alpha = results[(position, layer)][variable]["onehot"]["alpha"]
        coefficients = np.linalg.solve(gram + alpha * scale * np.eye(len(gram)), np.eye(VALUES)[values[train]])
        predicted = (centred_test @ centred_train.T @ coefficients).argmax(axis=1)
        truth = values[test]
        asked = asked_variable[test] == variable
        out[variable] = {
            "asked_accuracy": float(np.mean(predicted[asked] == truth[asked])),
            "not_asked_accuracy": float(np.mean(predicted[~asked] == truth[~asked])),
            "asked_rows": int(asked.sum()),
        }
    return out


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run = Path(argv[0])
    sites = [(spec.split(":")[0], int(spec.split(":")[1])) for spec in argv[1:]]
    print(json.dumps([breakdown(run, position, layer) for position, layer in sites], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
