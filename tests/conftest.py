import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "ti_tnn"))


def pytest_ignore_collect(collection_path, config):
    name = collection_path.name
    if not name.startswith("test_ti_tnn_native_") or name == "test_ti_tnn_native_ledger.py":
        return None
    if importlib.util.find_spec("torch") is None:
        return True
    if name in ("test_ti_tnn_native_data.py", "test_ti_tnn_native_structured.py", "test_ti_tnn_native_followup.py"):
        if importlib.util.find_spec("tokenizers") is None:
            return True
    return None
