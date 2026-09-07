import json
import subprocess
import sys
import numpy as np
import pytest
from universa_recurrent.examples import flow_example
from universa_recurrent.routing import select_constraint
from universa_recurrent.structures import CompiledConstraint
from universa_recurrent.cli import main


def test_router_uses_validation_and_selects():
    ex=flow_example()
    r=select_constraint(ex.candidates,ex.problem,ex.validation_measurement,ex.validation_observed)
    assert r.chosen.name == "balanced_flow"
    assert len(r.scores)==2


def test_ambiguous_structures_refused():
    ex=flow_example()
    c=ex.candidates[0]
    same=CompiledConstraint.compile("same_kernel",c.boundary * 2)
    r=select_constraint([c,same],ex.problem,ex.validation_measurement,ex.validation_observed)
    assert r.chosen is None
    assert 'refused' in r.reason


def test_cli_roundtrip_and_budget(tmp_path,capsys):
    path=tmp_path/'full.json'
    assert main(['demo','--trace','full','--output',str(path)]) == 0
    assert main(['verify',str(path)]) == 0
    assert main(['demo','--max-steps','1']) == 1
    assert main(['demo','--output',str(path)]) == 2
    assert 'PASS' in capsys.readouterr().out


def test_cli_bad_file(tmp_path):
    path=tmp_path/'bad.json'
    path.write_text('{')
    assert main(['verify',str(path)]) == 2
