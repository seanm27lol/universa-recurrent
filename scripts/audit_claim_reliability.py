#!/usr/bin/env python3
"""Independent saved-array quality audit. Imports NumPy, never producer/model code.

Usage: python scripts/audit_claim_reliability.py REPORTS.zip OUTPUT_DIRECTORY
Archive content is data only. NPZ loading forbids pickle and is size bounded.
Calibration order statistics and metric formulas are implemented here. Bootstrap
uses exactly the prespecified PCG64 seed/multinomial sampling distribution,
paired over common inputs and five fixed fits; chunks differ from the producer.
"""
import argparse, collections, hashlib, io, json, math, pathlib, statistics, sys, zipfile
import numpy as np

SEEDS = [6100,7100,8100,9100,10100]
MODELS = ['shared','fixed_depth_4','direct','gaussian_generator_reference']
TARGETS = [0.25,0.5,0.75,0.9]
class Audit:
    """Comparison counters belong to one run; numeric tolerance is absolute."""

    def __init__(self):
        self.counts = collections.Counter()
        self.differences = collections.defaultdict(float)
        self.mismatches = []

    def compare(self, want, got, path, group):
        if isinstance(want,dict):
            if not isinstance(got,dict) or set(want)!=set(got):
                self.mismatches.append({'path':path,'reason':'dictionary keys differ'})
                return
            for k,v in want.items():self.compare(v,got[k],path+'.'+k,group)
        elif isinstance(want,list):
            if not isinstance(got,list) or len(want)!=len(got):
                self.mismatches.append({'path':path,'reason':'list length differs'})
                return
            for i,v in enumerate(want):self.compare(v,got[i],f'{path}[{i}]',group)
        else:
            self.counts[group]+=1
            if isinstance(want,(int,float)) and not isinstance(want,bool):
                if not isinstance(got,(int,float)) or isinstance(got,bool) or not math.isfinite(got):
                    self.mismatches.append({'path':path,'expected':want,'reported':got});return
                delta=abs(want-got)
                self.differences[group]=max(self.differences[group],delta)
                # Integers/order-statistic thresholds compare exactly, general arithmetic 1e-12.
                tolerance=0 if isinstance(want,int) or path.endswith('.threshold') else 1e-12
                if delta>tolerance:self.mismatches.append({'path':path,'expected':want,'reported':got,'absolute_difference':delta})
            elif want!=got or (isinstance(want,bool) and type(got) is not bool):
                self.mismatches.append({'path':path,'expected':want,'reported':got})

def object_pairs(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key '+k)
        d[k]=v
    return d

def bounded_archive(z,max_size):
    members=z.infolist()
    if len(members)>100:raise ValueError('too many ZIP members')
    if len({v.filename for v in members})!=len(members):raise ValueError('duplicate ZIP member')
    if sum(v.file_size for v in members)>max_size:raise ValueError('oversized archive')
    for v in members:
        p=pathlib.PurePosixPath(v.filename)
        if p.is_absolute() or '..' in p.parts or v.file_size>30_000_000 or v.flag_bits&1:raise ValueError('unsafe ZIP member')

def read_arrays(payload, expected_shapes):
    # Check NPY headers before NumPy can allocate from their declared dimensions.
    with zipfile.ZipFile(io.BytesIO(payload)) as nested:
        bounded_archive(nested,50_000_000)
        if set(nested.namelist())!={k+'.npy' for k in expected_shapes}:
            raise ValueError('unexpected NPZ array inventory')
        for key, expected in expected_shapes.items():
            info=nested.getinfo(key+'.npy')
            with nested.open(info) as handle:
                version=np.lib.format.read_magic(handle)
                if version==(1,0):
                    shape, _, dtype=np.lib.format.read_array_header_1_0(handle)
                elif version==(2,0):
                    shape, _, dtype=np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError('unsupported NPY version')
                if shape!=expected or dtype.kind not in 'ifu' or dtype.hasobject:
                    raise ValueError('invalid NPY shape or dtype: '+key)
                size=math.prod(shape)*dtype.itemsize
                if size!=info.file_size-handle.tell():
                    raise ValueError('NPY payload size differs from its header')
    with np.load(io.BytesIO(payload),allow_pickle=False) as loaded:
        out={k:loaded[k] for k in loaded.files}
    for k,v in out.items():
        if v.dtype.kind not in 'ifu' or not np.isfinite(v).all():raise ValueError('invalid array '+k)
    return out

def expected_shapes(phase):
    n=5000 if phase=='calibration' else 20000
    result={k:(n,5) for k in ['observed','mask','truth']}
    result['labels']=(n,)
    result.update({m+'__probabilities':(n,2) for m in MODELS})
    if phase=='test':result.update({m+'__estimates':(n,5) for m in MODELS})
    return result

def probabilities(value):
    array=np.asarray(value)
    if array.dtype.kind not in 'ifu' or array.ndim!=2 or min(array.shape)<1 or array.shape[1]!=2:
        raise ValueError('probabilities must have nonempty shape [inputs, 2]')
    result=array.astype('float64')
    if (not np.isfinite(result).all() or (result<0).any() or (result>1).any()
            or not np.allclose(result.sum(axis=1),1,rtol=0,atol=1e-6)):
        raise ValueError('invalid probabilities')
    return result

def policies(cal):
    scores=probabilities(cal).max(axis=1)
    n=len(scores)
    result={}
    for q in TARGETS:
        required=math.ceil(q*n)
        cutoff=float(np.partition(scores,n-required)[n-required])
        count=int(np.count_nonzero(scores>=cutoff))
        result[str(q)]={'target_coverage':q,'threshold':cutoff,'calibration_count':n,
                       'calibration_claimed_count':count,'calibration_coverage':count/n}
    return result

def wilson(errors,n):
    if (type(errors) is not int or type(n) is not int or not 0<=errors<=n):
        raise ValueError('invalid binomial counts')
    if not n:return None
    z=statistics.NormalDist().inv_cdf(0.975)
    center=(errors+z*z/2)/(n+z*z)
    radius=z*math.sqrt(errors*(n-errors)/n+z*z/4)/(n+z*z)
    return [max(0.,center-radius) if errors else 0.,min(1.,center+radius) if errors<n else 1.]

def row_metrics(cal,test,estimate,truth,label):
    prob=probabilities(test)
    cal=probabilities(cal)
    n=len(prob)
    label=np.asarray(label)
    if label.dtype.kind not in 'iu' or label.shape!=(n,) or (label<0).any() or (label>=2).any():
        raise ValueError('labels must be an integer vector in [0, 2)')
    target=np.asarray(truth)
    estimate=np.asarray(estimate)
    if (target.dtype.kind not in 'ifu' or estimate.dtype.kind not in 'ifu'
            or target.shape!=(n,5) or estimate.shape!=(n,5)):
        raise ValueError('truth and estimates must have shape [inputs, 5]')
    target=target.astype('float64')
    estimate=estimate.astype('float64')
    if not np.isfinite(target).all() or not np.isfinite(estimate).all():
        raise ValueError('truth and estimates must be finite')
    wrong=np.argmax(prob,axis=1)!=label
    with np.errstate(over='ignore', invalid='ignore'):
        loss=np.mean((estimate-target)**2,axis=1)
    if not np.isfinite(loss).all():raise ValueError('derived squared errors must be finite')
    brier=np.zeros(n)
    for k in range(prob.shape[1]):brier+=(prob[:,k]-(label==k))**2
    nll=-np.log(np.clip(prob[np.arange(n),label],1e-15,None))
    result={'n_calibration':len(cal),'n_test':n,'n_classes':prob.shape[1],
            'top_accuracy':1.-int(wrong.sum())/n,'mse':math.fsum(loss)/n,
            'brier_score':math.fsum(brier)/n,'nll':math.fsum(nll)/n,'coverage_results':{}}
    frozen=policies(cal)
    for key,p in frozen.items():
        claim=prob.max(axis=1)>=p['threshold']
        count=int(np.count_nonzero(claim));errors=int(np.count_nonzero(claim & wrong))
        result['coverage_results'][key]={**p,'test_count':n,'claimed_count':count,'abstained_count':n-count,
            'wrong_claim_count':errors,'coverage':count/n,'wrong_claim_rate':errors/count if count else None,
            'wrong_claim_rate_all_inputs':errors/n,'wrong_claim_rate_wilson95':wilson(errors,count) if count else None}
    return result,{'claimed':prob.max(axis=1)>=frozen['0.75']['threshold'],'wrong':wrong,'loss':loss}

def variation(values):
    defined=[v for v in values if v is not None]
    if not values or any(not math.isfinite(v) for v in defined):raise ValueError('invalid fit metrics')
    complete=len(defined)==len(values)
    return {'per_fit':values,'defined_fits':len(defined),'total_fits':len(values),
            'mean':statistics.mean(values) if complete else None,
            'minimum':min(values) if complete else None,'maximum':max(values) if complete else None,
            'sample_sd':statistics.stdev(values) if complete and len(values)>1 else None}

def aggregate(rows):
    return {'n_fits':len(rows),'per_fit':rows,'descriptive_fit_variation':{
        **{k:variation([v[k] for v in rows]) for k in ['mse','top_accuracy','brier_score','nll']},
        'coverage_results':{str(q):{k:variation([v['coverage_results'][str(q)][k] for v in rows])
                                  for k in ['coverage','wrong_claim_rate']} for q in TARGETS}}}

def bootstrap(values, repeats=2000, seed=43000):
    # Each draw is a bootstrap sample of input indices, represented by counts;
    # all ten fitted models use this same count vector. No fit resampling.
    left=values['shared'];right=values['fixed_depth_4']
    if not left or len(left)!=len(right) or type(repeats) is not int or repeats<1:
        raise ValueError('invalid paired bootstrap fits or repeat count')
    if type(seed) is not int or seed<0:raise ValueError('invalid bootstrap seed')
    n=len(left[0]['loss']);fits=len(left)
    if not n:raise ValueError('bootstrap needs inputs')
    for item in left+right:
        for key in ['claimed','wrong','loss']:
            value=np.asarray(item[key])
            if value.shape!=(n,):raise ValueError('bootstrap arrays must align')
            if key!='loss' and value.dtype.kind!='b':raise ValueError('bootstrap masks must be Boolean')
            if key=='loss' and (value.dtype.kind not in 'ifu' or not np.isfinite(value).all() or (value<0).any()):
                raise ValueError('bootstrap losses must be finite and nonnegative')
    claimed=np.array([v['claimed'] for v in left+right],dtype='float64')
    erroneous=np.array([v['claimed'] & v['wrong'] for v in left+right],dtype='float64')
    mean_loss=np.mean([l['loss']-r['loss'] for l,r in zip(left,right)],axis=0)
    mean_cover=np.mean([l['claimed'].astype('float64')-r['claimed'] for l,r in zip(left,right)],axis=0)
    if not np.isfinite(mean_loss).all():raise ValueError('derived bootstrap loss difference is not finite')
    rng=np.random.Generator(np.random.PCG64(seed))
    draws={k:[] for k in ['wrong_claim_rate_difference','mse_difference','coverage_difference']}
    for start in range(0,repeats,25):
        weights=rng.multinomial(n,[1/n]*n,size=min(25,repeats-start)).astype('float64')
        denominator=weights@claimed.T
        numerator=weights@erroneous.T
        valid=(denominator>0).all(axis=1)
        rates=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        risk=np.average(rates[:,:fits],axis=1)-np.average(rates[:,fits:],axis=1)
        draws['wrong_claim_rate_difference'].extend(float(v) if ok else None for v,ok in zip(risk,valid))
        mse=weights@mean_loss/n
        coverage=weights@mean_cover/n
        if not np.isfinite(mse).all() or not np.isfinite(coverage).all():raise ValueError('bootstrap produced nonfinite linear metrics')
        draws['mse_difference'].extend(mse.tolist())
        draws['coverage_difference'].extend(coverage.tolist())
    denominator=claimed.sum(axis=1)
    rate=np.divide(erroneous.sum(axis=1),denominator,out=np.zeros_like(denominator),where=denominator>0)
    points={'wrong_claim_rate_difference':float(rate[:fits].mean()-rate[fits:].mean()) if (denominator>0).all() else None,
            'mse_difference':float(mean_loss.mean()),'coverage_difference':float(mean_cover.mean())}
    results={}
    for key,values in draws.items():
        finite=all(v is not None for v in values)
        results[key]={'estimate':points[key],'percentile95':np.quantile(values,[.025,.975],method='linear').tolist() if finite else None,
                     'defined_resamples':sum(v is not None for v in values),'undefined_resamples':sum(v is None for v in values)}
    return results,draws

def audit_archive(archive):
    """Audit the fixed 5,000/20,000-input v1 study; no execution/provenance proof."""
    archive = pathlib.Path(archive)
    audit = Audit()
    with archive.open('rb') as handle:archive_bytes=handle.read(100_000_001)
    if len(archive_bytes)>100_000_000:raise ValueError('compressed archive is oversized')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        bounded_archive(z,100_000_000)
        expected={'results/'+k for k in ['summary.json','manifest.json','calibration-lock.json']}
        expected.update(f'results/{phase}-{seed}.{suffix}' for phase in ['calibration','test']
                        for seed in SEEDS for suffix in ['json','npz'])
        if set(z.namelist())!=expected:raise ValueError('unexpected outer archive inventory')
        contents={i.filename:z.read(i) for i in z.infolist()}
    docs={pathlib.PurePosixPath(k).name:json.loads(v,object_pairs_hook=object_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
          for k,v in contents.items() if k.endswith('.json')}
    summary=docs['summary.json'];lock=docs['calibration-lock.json']
    audit.compare(SEEDS,summary['training_seeds'],'summary.training_seeds','study_constants')
    audit.compare(SEEDS,lock['training_seeds'],'lock.training_seeds','study_constants')
    audit.compare(5000,summary['calibration_inputs']['count'],'summary.calibration_count','study_constants')
    audit.compare(20000,summary['test_inputs']['count'],'summary.test_count','study_constants')
    audit.compare(True,summary['prespecified_protocol'],'summary.prespecified_protocol','study_constants')
    for name in ['summary.json','manifest.json']:
        audit.compare(False,docs[name]['smoke'],name+'.smoke','study_constants')
    if len(lock['calibrations'])!=5 or len(summary['per_fit'])!=5:raise ValueError('expected exactly five fitted reports')
    protocol=docs['manifest.json']['effective_protocol']
    for key,value in {'calibration_size':5000,'test_size':20000,'calibration_seed':41000,
                      'test_seed':42000,'coverages':TARGETS,'primary_coverage':.75,
                      'training_seeds':SEEDS,'models':MODELS,'primary_comparison':['shared','fixed_depth_4'],
                      'bootstrap_repeats':2000,'bootstrap_seed':43000}.items():
        audit.compare(value,protocol[key],'protocol.'+key,'study_constants')
    for key,value in {'left_model':'shared','right_model':'fixed_depth_4','target_coverage':.75,
                      'n_fits':5,'n_unique_test_inputs':20000,'n_resamples':2000,'seed':43000}.items():
        audit.compare(value,summary['aggregate']['primary_comparison'][key],'primary.'+key,'study_constants')
    rows={m:[] for m in MODELS};inputs={m:[] for m in MODELS};per_fit=[];ties=[]
    first_arrays={}
    for i,seed in enumerate(SEEDS):
        cal_doc=docs[f'calibration-{seed}.json'];test_doc=docs[f'test-{seed}.json']
        ca=read_arrays(contents[f'results/calibration-{seed}.npz'],expected_shapes('calibration'))
        ta=read_arrays(contents[f'results/test-{seed}.npz'],expected_shapes('test'))
        for path,report in [(f'calibration-{seed}',cal_doc),(f'test-{seed}',test_doc),
                            (f'lock[{i}]',lock['calibrations'][i]),(f'summary.per_fit[{i}]',summary['per_fit'][i])]:
            audit.compare(seed,report['training_seed'],path+'.training_seed','study_constants')
        for phase,arrays in [('calibration',ca),('test',ta)]:
            label=arrays['labels']
            if label.dtype.kind not in 'iu' or (label<0).any() or (label>=2).any():raise ValueError('invalid cohort labels')
            if i==0:first_arrays[phase]=arrays
            for k in ['observed','mask','truth','labels']:
                audit.compare(True,bool(np.array_equal(arrays[k],first_arrays[phase][k])),f'{phase}-{seed}.{k}.common','common_arrays')
            for k in [key for key in arrays if key.startswith('gaussian_generator_reference__')]:
                audit.compare(True,bool(np.array_equal(arrays[k],first_arrays[phase][k])),f'{phase}-{seed}.{k}.common','common_arrays')
        computed={}
        for m in MODELS:
            cal=ca[m+'__probabilities'];test=ta[m+'__probabilities'];estimate=ta[m+'__estimates']
            frozen=policies(cal)
            for path,reported in [(f'calibration-{seed}.policies',cal_doc['policies']),
                (f'test-{seed}.policies',test_doc['policies']),
                (f'lock.{seed}.policies',lock['calibrations'][i]['policies'])]:
                audit.compare(frozen,reported[m],path+'.'+m,'policy_numeric_leaves')
            calculated,paired=row_metrics(cal,test,estimate,ta['truth'],ta['labels'])
            computed[m]=calculated
            audit.compare(calculated,test_doc['analysis'][m],f'test-{seed}.analysis.{m}','test_metric_leaves')
            audit.compare(calculated,summary['per_fit'][i]['analysis'][m],f'summary.per_fit[{i}].analysis.{m}','summary_per_fit_metric_leaves')
            if m!='gaussian_generator_reference' or i==0:
                rows[m].append(calculated);inputs[m].append(paired)
            for q in TARGETS:
                p=frozen[str(q)]
                if p['calibration_claimed_count']>math.ceil(q*5000):
                    ties.append({'training_seed':seed,'model':m,**p,'test_coverage':calculated['coverage_results'][str(q)]['coverage']})
        per_fit.append({'training_seed':seed,'models':computed})
    recomputed={m:aggregate(rows[m]) for m in MODELS}
    audit.compare(recomputed,summary['aggregate']['models'],'summary.aggregate.models','aggregate_metric_leaves')
    paired,draws=bootstrap(inputs)
    for key,value in paired.items():audit.compare(value,summary['aggregate']['primary_comparison'][key],f'primary.{key}','bootstrap_numeric_leaves')
    primary=[];sweep=[]
    for model,items in rows.items():
        for q in TARGETS:
            a=[r['coverage_results'][str(q)] for r in items]
            risk=variation([v['wrong_claim_rate'] for v in a])
            entry={'model':model,'n_fits':len(items),'target_coverage':q,
                   'mean_coverage':statistics.mean(v['coverage'] for v in a),
                   'mean_wrong_claim_rate':risk['mean'],
                   'wrong_claim_rate_range':[risk['minimum'],risk['maximum']],
                   'wrong_counts':[v['wrong_claim_count'] for v in a],'claimed_counts':[v['claimed_count'] for v in a]}
            sweep.append(entry)
            if q==.75:
                primary.append({**entry,**{k:statistics.mean(v[k] for v in items) for k in ['mse','top_accuracy','brier_score','nll']}})
    differences=[]
    for i,seed in enumerate(SEEDS):
        l=rows['shared'][i];r=rows['fixed_depth_4'][i]
        left_risk=l['coverage_results']['0.75']['wrong_claim_rate']
        right_risk=r['coverage_results']['0.75']['wrong_claim_rate']
        differences.append({'training_seed':seed,
            'wrong_claim_rate_difference':left_risk-right_risk if left_risk is not None and right_risk is not None else None,
            'coverage_difference':l['coverage_results']['0.75']['coverage']-r['coverage_results']['0.75']['coverage'],
            'mse_difference':l['mse']-r['mse']})
    result={'audit_format':'universa.claim-reliability.independent-metrics-audit.v1',
        'input_archive':archive.name,'input_zip_sha256':hashlib.sha256(archive_bytes).hexdigest(),
        'method':'Independent formulas from saved finite numeric NPZ arrays. No producer analysis, neural inference, torch, or unpickling.',
        'numpy_version':np.__version__,'checks':dict(audit.counts),'total_scalar_comparisons':sum(audit.counts.values()),
        'max_absolute_difference_by_group':dict(audit.differences),'mismatches':audit.mismatches,'status':'PASS' if not audit.mismatches else 'FAIL',
        'distinct_model_fit_predictions':16,'per_fit_reports_recomputed':20,'test_input_count':20000,
        'primary_table':primary,'coverage_sweep':sweep,'primary_paired_bootstrap':paired,
        'primary_per_fit_differences':differences,'inclusive_tie_overshoots':ties,
        'aggregate':recomputed,'per_fit':per_fit,
        'limits':['Saved predictions are replayed arithmetically; neural weights and remote execution are not authenticated.',
                  'Five neural fits share the same 20000 test inputs; the Gaussian reference is one deterministic fit.',
                  'Intervals condition on fixed fitted models and fixed calibration policies; exclude training and calibration uncertainty.',
                  'No equivalence margin was prespecified; intervals spanning zero do not establish equivalence.',
                  'Target coverage is not necessarily achieved held-out coverage, especially with inclusive cutoff ties.',
                  'Policy fields are compared with the calibration lock; file hashes, source identity and execution chronology are outside this metrics audit.']}
    return result, draws


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path, help='New output directory; refuses to overwrite')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('output directory already exists; choose a new path')
        result, draws = audit_archive(args.archive)
        encoded={name:json.dumps(value,indent=2,allow_nan=False)+'\n'
                 for name,value in [('metrics-audit.json',result),('bootstrap-draws.json',draws)]}
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in encoded.items():
            with (args.output / name).open('x') as handle:
                handle.write(value)
        print(json.dumps({k: result[k] for k in ('status', 'checks', 'total_scalar_comparisons',
                         'max_absolute_difference_by_group', 'mismatches')}, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError, IndexError,
            zipfile.BadZipFile, EOFError) as exc:
        print(f'Audit error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
