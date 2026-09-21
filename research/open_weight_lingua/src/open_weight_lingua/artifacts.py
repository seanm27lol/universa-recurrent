"""Fresh run directories, safe numeric evidence and a reports-only inventory."""

import hashlib
import json
from pathlib import Path
import zipfile
import torch
from safetensors import safe_open
from safetensors.torch import save as serialize_tensors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def save_numeric(path: Path, values: dict[str, torch.Tensor]):
    if Path(path).suffix != ".safetensors":
        raise ValueError("numeric evidence must use safetensors")
    clean = {key: value.detach().cpu().contiguous() for key, value in values.items()}
    if any(not torch.isfinite(value).all() for value in clean.values()):
        raise ValueError("numeric evidence contains nonfinite values")
    with Path(path).open("xb") as stream:
        stream.write(serialize_tensors(clean))


def load_numeric(path: Path, *, max_bytes=64 * 1024 * 1024):
    path = Path(path)
    if path.suffix != ".safetensors" or path.stat().st_size > max_bytes:
        raise ValueError(
            "unsupported or oversized numeric evidence; pickle/NumPy objects forbidden"
        )
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        values = {name: stream.get_tensor(name) for name in stream.keys()}
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("numeric evidence contains nonfinite values")
    return values


class RunDirectory:
    """No implicit resumption. An existing run is an error, even if empty."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "raw").mkdir()

    def reports_zip(self):
        reports = sorted(
            path
            for path in self.path.glob("*")
            if path.is_file() and path.suffix in (".json", ".md")
        )
        retained = sorted(
            path for path in (self.path / "raw").glob("*") if path.is_file()
        )
        inventory = {
            "included": [
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in reports
            ],
            "retained_locally_not_in_zip": [
                {
                    "file": str(path.relative_to(self.path)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in retained
            ],
            "limits": "Raw vectors and full-vocabulary logits stay local; ZIP alone cannot replay KL. Hashes do not authenticate execution.",
        }
        write_json(self.path / "inventory.json", inventory)
        with zipfile.ZipFile(
            self.path / "reports.zip", "x", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in [*reports, self.path / "inventory.json"]:
                archive.write(path, arcname=path.name)
