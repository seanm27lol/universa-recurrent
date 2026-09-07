import importlib.util
import json
from pathlib import Path


def test_exploratory_benchmark(tmp_path):
    path=Path(__file__).resolve().parents[1]/'experiments'/'benchmark.py'
    spec=importlib.util.spec_from_file_location('benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out=tmp_path/'timings.json'
    assert module.main(['--output',str(out),'--repeats','1'])==0
    data=json.loads(out.read_text())
    assert len(data['rows'])==9
    assert all(r['status']=='checked' and r['total_ms']>=0 for r in data['rows'])
    assert data['status']=='exploratory_microbenchmark_not_preregistered'
