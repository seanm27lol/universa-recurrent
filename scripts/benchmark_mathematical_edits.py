"""Do typed mathematical edits preserve their computational effect through a codec?

The command names fields whose meanings are known by construction. This is NOT
learned English or discovered neuron semantics. Direct numerical edits are the
reference; a wrong-candidate edit is a specificity control. No weights are trained.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import os
import random
import statistics
import subprocess
import sys
import numpy as np
import torch

import benchmark_state_continuation as prior
from continuation_codec import canonical
from overflow_state_codec import OverflowStateCodec
from mathematical_edits import (render, direct_edit, apply_command, decode_and_edit,
                                wrong_candidate, edit_plan, primitive_error)
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json, dataset_fingerprint
from universa_recurrent.neural.retention_study import configure, compare_outputs
from universa_recurrent.neural.train import sha256_file

FORMAT='universa-recurrent.mathematical-edits.v1'
FORBIDDEN={61000,62000,71000,72000,46000,32000,31000,36000,12000,9000}
SCOPE={
 'question':'Does a typed edit to a decoded 16-bit numerical description preserve the effect of the same direct float32 state edit?',
 'not_claimed':'No learned language interpreter, discovered semantic variable, formal causal abstraction, correct structure, speedup or total-memory compression.',
 'decoder':'Receives state payloads, frozen codec metadata and a typed command only. Original context remains only with the resumed solver.',
 'commands':'Cycle += +/- one eighth of its frozen calibration range; one current evidence logit += +/- log(2); identity; common-logit shift.',
 'controls':'Raw-state restoration; parser/raw-numeric edit equality; unchanged state; same edit applied to the WRONG candidate. Neutral common-logit shift is measured, not assumed bitwise invariant.',
 'direction':'The immediate cycle-vector / softmax effect is known algebraically. No direction is predicted for the FINAL estimate.',
 'comparison':'Effects relative to each path own no-edit baseline, as well as absolute agreement with raw edited output. Command bytes and baseline codec/context costs count separately.',
 'sampling':'One common matched-distribution 512-input cohort, repeated across checkpoints, cuts and edits; not independent extra problems.',
 'bounds':'Changing a variable changes the experimental task. Task accuracy after arbitrary edits is not an improvement objective. Nonfinite continuations remain failures.',
}


def identity():
    result=prior.source();root=Path(__file__).resolve().parent
    result['edit_scripts']={n:sha256_file(root/n) for n in
        ('benchmark_mathematical_edits.py','mathematical_edits.py','overflow_state_codec.py')}
    return result


def cut_points(depth):
    if type(depth) is not int or depth<2:raise ValueError('depth must allow continuation')
    return sorted({max(1,depth//2),depth-1})


def metrics(output, direct, baseline, direct_baseline):
    """Difference of effects isolates codec distortion from base-state error."""
    x=output['estimate'].detach().cpu().numpy().astype(float)
    b=baseline['estimate'].numpy().astype(float)
    delta=x-b
    if direct is None:
        return dict(effect_mse=float((delta**2).mean()),comparison_available=False)
    y=direct['estimate'].numpy().astype(float)
    true_delta=y-direct_baseline['estimate'].numpy().astype(float)
    return dict(comparison_available=True,
        final_mse_to_direct=float(((x-y)**2).mean()),
        worst_component_difference=float(np.abs(x-y).max()),
        effect_mse=float((delta**2).mean()),
        direct_effect_mse=float((true_delta**2).mean()),
        effect_disagreement_mse=float(((delta-true_delta)**2).mean()),
        changed_claims_vs_direct=int((output['claim_index'].cpu()!=direct['claim_index']).sum()),
        changed_claims_vs_own_baseline=int((output['claim_index'].cpu()!=baseline['claim_index']).sum()),
        changed_routes_vs_direct=int((output['top_route'].cpu()!=direct['top_route']).sum()))


def save_output(arrays,key,output):
    for field in ('estimate','probabilities','claim_index','top_route'):
        arrays[key+'__'+field]=output[field].detach().cpu().numpy().copy()


@torch.inference_mode()
def worker(args):
    if args.test_seed in FORBIDDEN or args.n<2:raise ValueError('new seed and at least two inputs required')
    torch.set_num_threads(1);execution=configure(True);start_source=identity()
    paths=(args.checkpoint,args.claim_calibration,args.previous_report)
    hashes=tuple(sha256_file(p) for p in paths);cp_sha,claim_sha,old_sha=hashes
    old=read_json(args.previous_report)
    if (old.get('format')!=prior.FORMAT+'.worker' or old.get('all_restoration_gates_passed') is not True
        or old['checkpoint_sha256']!=cp_sha or old['claim_calibration_sha256']!=claim_sha):
        raise ValueError('original continuation report/reference mismatch')
    engines,meta,device,reserved=load_estimators(args.checkpoint,args.device)
    claims=read_json(args.claim_calibration)
    if meta['checkpoint_sha256']!=cp_sha or claims['checkpoint_sha256']!=cp_sha:
        raise ValueError('checkpoint/calibration mismatch')
    reserved=set(reserved)|{claims['data']['seed'],old['calibration_seed'],old['test_seed'],old['training_seed']}
    if 'test_seed_reserved' in claims:reserved.add(claims['test_seed_reserved'])
    if args.test_seed in reserved:
        raise ValueError('input seed overlaps prior training/calibration')
    data=StructuredFlowDataset(args.n,seed=args.test_seed,**{k:meta['training'][k] for k in ('noise_std','observe_probability')})
    joined=np.concatenate((data.observed.numpy(),data.mask.numpy()),-1)
    if len({x.tobytes() for x in joined})!=args.n:raise ValueError('non-distinct inputs')
    arrays={k:v.numpy() for k,v in dict(observed=data.observed,mask=data.mask,truth=data.truth,labels=data.label).items()}
    x,m=data.observed.to(device),data.mask.to(device);by_name={e.name:e for e in engines}
    rows=[];gates=[];sites=[]
    for name in args.models:
        engine=by_name[name];model=engine.model.eval();d=model.config.latent_dim
        if claims['model_sources'][name]!=cp_sha:raise ValueError('wrong claim source')
        policy=ClaimPolicy(**claims['models'][name]['policy'])
        names=meta['names'];q=model.bases.detach().cpu().numpy()
        h=model.rollout(x,m)
        reference=prior.as_cpu(construct_output(h['states'][-1],h['route_probabilities'][-1],policy,'mixture_with_claim'));del h
        save_output(arrays,name+'__original',reference)
        for cut in cut_points(model.config.steps):
            key=f'{name}__cut{cut}';point=prior.pause(model,x,m,cut);original=point.vector()
            raw,used=prior.continue_from(model,point,original,policy)
            gates.append(dict(model=name,cut=cut,checks=compare_outputs(raw,reference)))
            if used!=model.config.steps-cut:raise ValueError('wrong remaining budget')
            base=old['codecs'][f'{name}__cut{cut}__named_16bit']
            expected_site=dict(checkpoint_sha256=cp_sha,model=name,cut=cut,depth=model.config.steps,
                num_structures=model.config.num_structures,latent_dim=d,names=names,calibration_sha256=old['calibration_sha256'])
            if base['site']!=expected_site or base['calibration_n']!=old['calibration_n']:
                raise ValueError('frozen site mismatch')
            codec=OverflowStateCodec.from_manifest(base);payloads,fb=codec.encode(original);decoded=codec.decode(payloads)
            decoded_base,used=prior.continue_from(model,point,decoded,policy);decoded_base=prior.as_cpu(decoded_base)
            arrays[key+'__original_dynamic']=original;arrays[key+'__decoded_dynamic']=decoded;arrays[key+'__fallback_mask']=fb
            save_output(arrays,key+'__decoded_baseline',decoded_base)
            commands=edit_plan(base)
            sites.append(dict(model=name,cut=cut,codec=codec.manifest(),storage=codec.storage(payloads),
                              retained_context_bytes_per_input=point.retained_bytes_per_example(),commands=[render(e,names,d) for e in commands]))
            for index,edit in enumerate(commands):
                command=render(edit,names,d);raw_edit=direct_edit(original,edit,names,d)
                if not np.array_equal(raw_edit,apply_command(original,command,names,d)):
                    raise ValueError('raw parser/numerical edit mismatch')
                typed=decode_and_edit(payloads,codec,command)
                # Do not re-quantize after the edit: the descriptor is base payload + command.
                variants=[('direct_float32',raw_edit),('decoded_named_edit',typed)]
                wrong=wrong_candidate(edit,names,d)
                if wrong is not None:
                    variants.append(('wrong_candidate_control',decode_and_edit(payloads,codec,render(wrong,names,d))))
                direct=None
                for mode,state in variants:
                    array_key=key+f'__edit{index}__'+mode
                    arrays[array_key+'__edited_dynamic']=state.copy()
                    try:out,updates=prior.continue_from(model,point,state,policy)
                    except prior.NumericalContinuationError as error:
                        rows.append(dict(model=name,cut=cut,edit_index=index,command=command,mode=mode,status='nonfinite_continuation',metrics=None,failure=str(error)));continue
                    if updates!=model.config.steps-cut:raise ValueError('edit changed remaining budget')
                    out=prior.as_cpu(out)
                    if mode=='direct_float32':direct=out
                    own_base=reference if mode=='direct_float32' else decoded_base
                    measured=metrics(out,direct,own_base,reference)
                    applied=wrong if mode=='wrong_candidate_control' else edit
                    primitive=primitive_error(original if mode=='direct_float32' else decoded,state,applied,q,names,d)
                    if mode=='decoded_named_edit' and fb.any() and direct is not None:
                        compare_outputs({k:v[fb] for k,v in out.items()},{k:v[fb] for k,v in direct.items()})
                    save_output(arrays,array_key,out)
                    rows.append(dict(model=name,cut=cut,edit_index=index,edit=asdict(edit),command=command,
                        applied_command=render(applied,names,d),command_utf8_bytes=len(render(applied,names,d).encode()),
                        command_scope='one common command per input batch, or this many extra bytes per standalone descriptor',
                        mode=mode,status='finite',n=args.n,executed_updates=updates,metrics=measured,
                        immediate_rule_errors=primitive))
            if original.tobytes()!=point.vector().tobytes():raise ValueError('pause state mutated')
            print(f'{name} cut {cut}: {len(commands)} typed edits and controls completed',flush=True)
    if tuple(sha256_file(p) for p in paths)!=hashes or identity()!=start_source:raise ValueError('source/reference changed')
    npz=args.worker_output.with_suffix('.npz')
    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(format=FORMAT+'.worker',source=start_source,execution=execution,checkpoint_sha256=cp_sha,
        claim_calibration_sha256=claim_sha,previous_report_sha256=old_sha,training_seed=meta['training']['seed'],
        test_seed=args.test_seed,n=args.n,dataset_sha256=dataset_fingerprint(data),device=str(device),
        rows=rows,sites=sites,raw_gates=gates,all_raw_gates_passed=True,no_training=True,no_refitting=True,
        per_example_file=npz.name,per_example_sha256=sha256_file(npz),scope=SCOPE)


def parent(args):
    if args.output_dir.exists():raise ValueError('refusing existing output directory')
    jobs=[]
    for cp in sorted(args.replication_dir.glob('weights-*.pt')):
        seed=int(cp.stem.removeprefix('weights-'));claim=args.replication_dir/f'study-{seed}/calibration.json';old=args.previous_dir/f'seed-{seed}.json'
        if not claim.is_file() or not old.is_file():raise ValueError('missing saved reference for '+str(seed))
        jobs.append((seed,cp,claim,old))
    if not jobs or len({sha256_file(x[1]) for x in jobs})!=len(jobs):raise ValueError('distinct frozen checkpoints required')
    random.Random(84000).shuffle(jobs);initial=identity();args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'PLAN.json',dict(format=FORMAT+'.plan',test_seed=args.test_seed,n=args.n,models=args.models,
        job_order=[x[0] for x in jobs],source=initial,scope=SCOPE,primary='decoded_named_edit versus direct_float32 effects; wrong-candidate control',no_training=True))
    reports=[];cohort=None
    for seed,cp,claim,old in jobs:
        out=args.output_dir/f'seed-{seed}.json'
        cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp.resolve()),'--claim-calibration',str(claim.resolve()),
             '--previous-report',str(old.resolve()),'--worker-output',str(out.resolve()),'--device',args.device,
             '--n',str(args.n),'--test-seed',str(args.test_seed),'--models',*args.models]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w=read_json(out)
        if w['source']!=initial or w['training_seed']!=seed or w['checkpoint_sha256']!=sha256_file(cp):raise ValueError('worker identity mismatch')
        if w['claim_calibration_sha256']!=sha256_file(claim) or w['previous_report_sha256']!=sha256_file(old):raise ValueError('reference changed')
        if cohort is not None and w['dataset_sha256']!=cohort:raise ValueError('cohorts differ')
        cohort=w['dataset_sha256'];reports.append(w)
    failed=sum(r['status']!='finite' for w in reports for r in w['rows'])
    result=dict(format=FORMAT+'.complete',checkpoint_runs=len(reports),test_seed=args.test_seed,n=args.n,source=initial,
        worker_files=[f'seed-{w["training_seed"]}.json' for w in reports],all_raw_gates_passed=all(w['all_raw_gates_passed'] for w in reports),
        failed_cases=failed,dataset_sha256=cohort,scope=SCOPE,
        note='Inspect per-example effect fidelity and wrong-name controls. Known-field parser correctness is not discovery of semantic mechanisms.')
    write_json(args.output_dir/'summary.json',result);print('COMPLETE:',args.output_dir/'summary.json',flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('replication-dir','previous-dir','output-dir','checkpoint','claim-calibration','previous-report','worker-output'):p.add_argument('--'+k,type=Path)
    p.add_argument('--models',nargs='+',choices=['shared','fixed_depth_4'],default=['shared','fixed_depth_4'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--n',type=prior.positive,default=512)
    p.add_argument('--test-seed',type=prior.natural,default=83000)
    args=p.parse_args(argv)
    try:
        if args.n<2 or args.n>4096 or len(set(args.models))!=len(args.models) or args.test_seed in FORBIDDEN:raise ValueError('invalid size/models/new seed')
        if args.worker_output is not None:
            if not all((args.checkpoint,args.claim_calibration,args.previous_report)):raise ValueError('worker requires reference files')
            if args.worker_output.exists() or args.worker_output.with_suffix('.npz').exists():raise ValueError('refusing existing worker outputs')
            write_json(args.worker_output,worker(args))
        else:
            if not all((args.replication_dir,args.previous_dir,args.output_dir)):raise ValueError('supply replication, previous and output directories')
            parent(args)
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,FloatingPointError,subprocess.CalledProcessError) as e:
        print('Error:',e,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
