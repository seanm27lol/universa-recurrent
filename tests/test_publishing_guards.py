"""Offline tests of publishing guardrails. No GitHub requests are made."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('mode',['wrong_user','exists'])
def test_publish_refuses_before_writing(tmp_path,mode):
    root=tmp_path/'project'
    scripts=root/'scripts'
    scripts.mkdir(parents=True)
    original=Path(__file__).resolve().parents[1]/'scripts'/'publish.sh'
    shutil.copyfile(original,scripts/'publish.sh')
    bins=tmp_path/'bin'
    bins.mkdir()
    gh=bins/'gh'
    gh.write_text('#!/usr/bin/env bash\n'
                  'case "$1 $2" in\n'
                  '"auth status") exit 0;;\n'
                  '"api --hostname") echo "${TEST_LOGIN}"; exit 0;;\n'
                  '"repo view") exit 0;;\n'
                  '*) echo unexpected-gh-action >&2; exit 9;;\n'
                  'esac\n')
    gh.chmod(0o755)
    git=bins/'git'
    git.write_text('#!/usr/bin/env bash\necho unexpected-git-write >&2; exit 9\n')
    git.chmod(0o755)
    env=os.environ.copy()
    env['PATH']=str(bins)+os.pathsep+env['PATH']
    env['TEST_LOGIN']='someone_else' if mode=='wrong_user' else 'seanm27lol'
    result=subprocess.run(['bash',str(scripts/'publish.sh')],env=env,text=True,capture_output=True)
    assert result.returncode==1
    assert 'Refusing' in result.stderr
    assert 'unexpected' not in result.stderr
    assert not (root/'.git').exists()
