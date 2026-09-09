"""Report accounting tests, not tests of model quality or remote provenance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from zipfile import ZipFile
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_replication.py'
spec = importlib.util.spec_from_file_location('replication_audit', SCRIPT)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixture_reports():
    reports = {'results/PLAN.json': {'training_seeds':[10,20], 'test_seed':100,
        'calibration_seed':200,'training_data_seeds':[11,12,21,22]}}
    all_rows = []
    for seed, shift in ((10,0.0),(20,0.01)):
        prefix = f'results/study-{seed}/'
        cal = {'data':{'seed':200,'sha256':'cal-cohort'},'checkpoint_sha256':str(seed)}
        reports[prefix+'calibration.json'] = cal
        rows=[]
        for model,error in (('shared',0.1),('direct',0.2)):
            for mode in ('mixture','mixture_with_claim','hard'):
                row={'model':model,'output_mode':mode,'estimate_mse':error+shift,'n':4,'coverage':None}
                if mode!='mixture':
                    count=4 if mode=='hard' else 3
                    row.update(coverage=count/4,claimed_n=count,wrong_claim_n=1,
                        abstained_n=4-count,wrong_claim_rate_all=.25,wrong_claim_rate_claimed=1/count)
                rows.append(row)
        all_rows.append(rows)
        data={'seed':100,'sha256':'test-cohort','n':4,'batch_size':2}
        reports[prefix+'eval.json']={'dataset':data,'rows':rows,
            'environment':{'package_version':'0.5.0','git_commit':'a'*40},
            'calibration_sha256':hashlib.sha256(json.dumps(cal).encode()).hexdigest()}
        reports[prefix+'benchmark.json']={'dataset':copy.deepcopy(data),'repeats':3,
            'rows':[{**r,'median_ms_for_n_examples':2.,'samples_ms':[1.,2.,3.]} for r in rows]}
        reports[prefix+'PLAN.json']={'training_seed':seed,'checkpoint_sha256':str(seed),
            'test_seed':100,'calibration_seed':200,'reserved_data_seeds':[seed+1,seed+2]}
        reports[prefix+'COMPLETED.json']={'verified':True,'checkpoint_sha256':str(seed),'rows':6,'timed_rows':6}
        reports[prefix+'verification.json']={'accepted':True}
    summaries=[]
    for i,row in enumerate(all_rows[0]):
        values=[r[i]['estimate_mse'] for r in all_rows]
        summaries.append({'method':row['model']+'/'+row['output_mode'],
            'estimate_mse_across_training_seeds':{'mean':statistics.mean(values),
                'sample_sd':statistics.stdev(values),'per_seed':values}})
    reports['results/summary.json']={'training_seeds':[10,20],'replicates':2,
        'test_sha256':'test-cohort','rows':summaries}
    return reports


def save(tmp_path, reports):
    path=tmp_path/'reports.zip'
    with ZipFile(path,'w') as z:
        for name,data in reports.items():
            z.writestr(name,json.dumps(data))
    return path


def test_valid_reports_and_scope(tmp_path):
    result=audit_module.audit(save(tmp_path,fixture_reports()))
    assert result['report_consistency_passed']
    shared=next(r for r in result['rows'] if r['method']=='shared/mixture')
    assert shared['mse_mean']==pytest.approx(.105)
    assert shared['mean_of_run_median_ms']==2.
    assert 'no model weights' in result['scope']


@pytest.mark.parametrize('case', ['median','test_cohort','missing_timing','duplicate_method',
    'not_accepted','wrong_denominator','summary_mean','source','seed_overlap','calibration_hash'])
def test_inconsistent_report_rejected(tmp_path,case):
    r=fixture_reports(); p='results/study-10/'
    if case=='median': r[p+'benchmark.json']['rows'][0]['median_ms_for_n_examples']=9
    if case=='test_cohort': r[p+'benchmark.json']['dataset']['sha256']='changed'
    if case=='missing_timing': r[p+'benchmark.json']['rows'].pop()
    if case=='duplicate_method': r[p+'eval.json']['rows'].append(r[p+'eval.json']['rows'][0])
    if case=='not_accepted': r[p+'verification.json']['accepted']=False
    if case=='wrong_denominator':
        r[p+'eval.json']['rows'][1]['wrong_claim_rate_claimed']=.25
        r[p+'benchmark.json']['rows'][1]['wrong_claim_rate_claimed']=.25
    if case=='summary_mean': r['results/summary.json']['rows'][0]['estimate_mse_across_training_seeds']['mean']=100
    if case=='source': r[p+'eval.json']['environment']['package_version']='0.4.1'
    if case=='seed_overlap': r['results/PLAN.json']['training_data_seeds'].append(100)
    if case=='calibration_hash': r[p+'calibration.json']['data']['seed']=999
    with pytest.raises(ValueError): audit_module.audit(save(tmp_path,r))


@pytest.mark.parametrize('data', [b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b'[]'])
def test_strict_json(data):
    with pytest.raises(ValueError): audit_module.read_json(data)
