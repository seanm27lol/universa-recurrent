"""Build receipts in small stacks; investigate pauses without hiding their cost.

All modes keep the same GPU batch, weights, JSON bytes and batch checker. Only
host-side conversion is chunked. GC telemetry is a separate diagnostic pass;
normal timings never disable garbage collection or subtract pauses.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

import benchmark_batched_pipeline as previous
import benchmark_verified_pipeline as original
from batch_receipt_verifier import BatchPreparedVerifier, expected_inputs, generate_payloads
from benchmark_verifier_setup import PreparedVerifier, digest, positive, stats
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json, dataset_fingerprint
from universa_recurrent.neural.retention_lingua import FIELDS, loads
from universa_recurrent.neural.retention_study import compare_outputs, configure

FORMAT = 'universa-recurrent.bounded-receipts.v1'
MODES = ('per_record', 'bulk', 'chunk16', 'chunk64')
SCOPE = {
    'changed': 'Host tensor-to-Python conversion only; chunk sizes 16 and 64 do NOT split GPU inference.',
    'baseline': 'Original per-record generator with existing batch checker; current bulk generator is a second control.',
    'fixed': 'Frozen checkpoints, saved policies, one inference batch, exact individual JSON bytes, per-record checks and decisions.',
    'timed': 'CPU inputs through transfer, inference, host receipt generation/JSON and all bound/request-input checks, with GPU synchronization.',
    'excluded': 'Model loading, verifier preparation, synthesis, report writing, process startup, network/queueing, post-timing output comparison and rejection trials.',
    'gc': 'GC remains enabled at interpreter thresholds. No forced collection, freeze, threshold tuning, or pause subtraction.',
    'diagnostics': 'Separate callback-instrumented repetitions are not pooled into primary latency statistics; observer allocations may perturb GC.',
    'repetition': 'Same 256-problem cohort reused for a performance test; nested request prefixes and checkpoints are not new independent inputs.',
    'limits': 'Numerical checks do not certify learned transitions, route truth, semantics, or execution authenticity. No speedup assumed.',
}


def generate_bounded(engine, metadata, policy, observed_cpu, mask_cpu, output,
                     history, calibration_sha, *, chunk_size=16):
    """Copy tensors to CPU once; materialize at most chunk_size Python records.

    Retain the historical individual-byte format. The full CPU tensors and final
    byte payload list still occupy memory; this is NOT constant total memory.
    """
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise ValueError('chunk_size must be an integer in [1,4096]')
    if observed_cpu.ndim != 2 or mask_cpu.shape != observed_cpu.shape:
        raise ValueError('observed/mask must have matching [batch,ambient] shapes')
    count = len(observed_cpu)
    if not 1 <= count <= 4096:
        raise ValueError('record count must be in [1,4096]')
    fields = ('candidate_states', 'probabilities', 'estimate', 'claim_mask', 'top_route')
    if any(len(output[key]) != count for key in fields):
        raise ValueError('output batch length mismatch')
    if history is not None and any(history[key].shape[1] != count for key in FIELDS):
        raise ValueError('history batch length mismatch')
    host = {key: output[key].detach().cpu() for key in fields}
    x, mask = observed_cpu.detach().cpu(), mask_cpu.detach().cpu()
    hist = None if history is None else {key: history[key].detach().cpu() for key in FIELDS}
    view = SimpleNamespace(name=engine.name, depth=engine.depth,
                           bases=engine.bases.detach().cpu() if hist is not None else None)
    meta = dict(metadata)
    boundary = meta['boundaries']
    if hasattr(boundary, 'detach'):
        meta['boundaries'] = boundary.detach().cpu()
    payloads = []
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        block = {key: value[start:stop] for key, value in host.items()}
        trace = None if hist is None else {key: value[:, start:stop] for key, value in hist.items()}
        payloads.extend(generate_payloads(view, meta, policy, x[start:stop], mask[start:stop],
                                         block, trace, calibration_sha))
    if len(payloads) != count:
        raise ValueError('incomplete generation')
    return payloads


class GCProbe:
    """Optional diagnostics only. Never changes the GC policy or suppresses pauses."""
    def __init__(self):
        self.stage = 'outside_request'
        self.events = []
        self.pending = None
        self.callback = self._callback

    def _callback(self, phase, info):
        now = time.perf_counter_ns()
        if phase == 'start':
            self.pending = (now, int(info['generation']), self.stage)
        elif phase == 'stop' and self.pending is not None:
            start, generation, stage = self.pending
            self.pending = None
            self.events.append((start, now, generation, stage,
                                int(info.get('collected', 0)), int(info.get('uncollectable', 0))))

    def __enter__(self):
        gc.callbacks.append(self.callback)
        return self

    def __exit__(self, *args):
        gc.callbacks.remove(self.callback)

    def observations(self, start, end):
        events = [dict(start_offset_ms=(a-start)/1e6, duration_ms=(b-a)/1e6,
                       generation=g, stage=s, collected=c, uncollectable=u)
                  for a,b,g,s,c,u in self.events if a >= start and b <= end]
        self.events.clear()
        return {'events': events, 'observed_gc_ms': sum(e['duration_ms'] for e in events)}


def scripts_identity():
    return {**previous.scripts_identity(), Path(__file__).name: digest(Path(__file__))}


@torch.inference_mode()
def request(engine, meta, policy, obs, mask, kind, mode, context, batched,
            cal_sha, device, expected, probe=None):
    if mode not in MODES:
        raise ValueError('unknown generation mode')
    original.sync(device)
    start = time.perf_counter_ns()
    stages = {}
    def mark(stage):
        if probe is not None:
            probe.stage = stage
        return time.perf_counter_ns()
    t=mark('input_transfer')
    x,m=obs.to(device),mask.to(device)
    original.sync(device);stages['input_transfer']=original.elapsed(t)
    t=mark('inference')
    output,history,_=original.infer(engine,policy,x,m,kind)
    original.sync(device);stages['inference']=original.elapsed(t)
    t=mark('receipt_generation_json')
    args=(engine,meta,policy,obs,mask,output,history,cal_sha)
    if mode=='per_record':
        payloads=original.generate_receipts(*args)
    elif mode=='bulk':
        payloads=generate_payloads(*args)
    else:
        payloads=generate_bounded(*args,chunk_size=int(mode.removeprefix('chunk')))
    original.sync(device);stages['receipt_generation_json']=original.elapsed(t)
    t=mark('parse_bound_check')
    result=batched.verify_payloads(payloads,expected_count=len(expected),expected=expected)
    original.sync(device);stages['parse_bound_check']=original.elapsed(t)
    end=time.perf_counter_ns()
    if probe is not None:
        probe.stage='outside_request'
    if not result['accepted'] or result['checked_count']!=len(obs):
        raise ValueError('a receipt was rejected or missing')
    sample=dict(total_ms=(end-start)/1e6,stages_ms=stages,
                checked_count=result['checked_count'],accepted_count=result['accepted_count'],
                scalar_fallback_records=result['scalar_fallback_records'])
    if probe is not None:
        sample['gc']=probe.observations(start,end)
    return sample,payloads,output


def worker(args):
    torch.set_num_threads(args.cpu_threads)
    execution=configure(True)
    if not gc.isenabled():
        raise ValueError('GC must remain enabled')
    gc_before={'enabled':gc.isenabled(),'thresholds':list(gc.get_threshold()),
               'counts':list(gc.get_count()),'frozen_objects':gc.get_freeze_count()}
    callbacks_before=list(gc.callbacks)
    cp_sha,cal_sha=digest(args.checkpoint),digest(args.calibration)
    source,identities=original.source_identity(),scripts_identity()
    engines,meta,device,reserved=load_estimators(args.checkpoint,args.device)
    artifact=read_json(args.calibration)
    if args.test_seed in reserved|{artifact['data']['seed']}:
        raise ValueError('test seed overlaps training/calibration')
    if meta['checkpoint_sha256']!=cp_sha or artifact['checkpoint_sha256']!=cp_sha:
        raise ValueError('artifact mismatch')
    context=PreparedVerifier.prepare(args.checkpoint,args.calibration)
    batched=BatchPreparedVerifier(context)
    settings=meta['training']
    data=StructuredFlowDataset(max(args.record_counts),seed=args.test_seed,
        noise_std=settings['noise_std'],observe_probability=settings['observe_probability'])
    identity=original.input_identity(data.observed,data.mask)
    if identity['distinct_observed_mask_count']!=len(data):
        raise ValueError('benchmark inputs must be distinct')
    available={e.name:e for e in engines}
    if not set(args.models)<=set(available):raise ValueError('missing controls')
    rng=random.Random(args.order_seed+settings['seed'])
    cases=[(name,count,kind) for name in args.models for count in args.record_counts
           for kind in (('endpoint',) if name=='direct' else ('endpoint','trajectory'))]
    rng.shuffle(cases)
    rows=[]
    for name,count,kind in cases:
        engine=available[name];policy=ClaimPolicy(**artifact['models'][name]['policy'])
        obs,mask=data.observed[:count],data.mask[:count]
        expected=expected_inputs(obs,mask,name,kind)
        _,canonical,reference=request(engine,meta,policy,obs,mask,kind,'per_record',
            context,batched,cal_sha,device,expected)
        reference={k:v.detach().cpu() for k,v in reference.items()}
        safety=previous.rejection_tests(canonical,expected,context,batched)
        def run(mode,probe=None):
            sample,payloads,out=request(engine,meta,policy,obs,mask,kind,mode,
                context,batched,cal_sha,device,expected,probe)
            # Outside timing. Keep canonical bytes, NOT a heap of parsed histories.
            compare_outputs(out,reference)
            if payloads!=canonical:raise ValueError('canonical individual bytes changed')
            return sample
        for mode in MODES:
            for _ in range(args.warmup):run(mode)
        samples={mode:[] for mode in MODES};orders=[]
        for _ in range(args.repeats):
            order=list(MODES);rng.shuffle(order);orders.append(order)
            for mode in order:samples[mode].append(run(mode))
        diagnostic={mode:[] for mode in MODES};diagnostic_orders=[]
        # Separate instrumented pass; these numbers NEVER enter primary timings.
        with GCProbe() as probe:
            for _ in range(args.diagnostic_repeats):
                order=list(MODES);rng.shuffle(order);diagnostic_orders.append(order)
                for mode in order:diagnostic[mode].append(run(mode,probe))
        rows.append(dict(model=name,retention=kind,record_count=count,
            canonical_sha256=original.batch_digest(canonical),serialized_bytes=sum(map(len,canonical)),
            identical_outputs_and_bytes=True,rejection_tests=safety,
            samples=samples,mode_order_each_repeat=orders,
            timings={m:stats([s['total_ms'] for s in v]) for m,v in samples.items()},
            diagnostics=diagnostic,diagnostic_order_each_repeat=diagnostic_orders))
        print(f'Seed {settings["seed"]}: {name}/{kind}/{count}: all four generators agree',flush=True)
        del canonical,reference
    if (gc.isenabled()!=gc_before['enabled'] or list(gc.get_threshold())!=gc_before['thresholds']
            or gc.callbacks!=callbacks_before or gc.get_freeze_count()!=gc_before['frozen_objects']):
        raise ValueError('GC settings or callbacks changed')
    if (digest(args.checkpoint)!=cp_sha or digest(args.calibration)!=cal_sha
            or scripts_identity()!=identities or original.source_identity()!=source):
        raise ValueError('source/reference changed during run')
    return dict(format=FORMAT+'.worker',training_seed=settings['seed'],checkpoint_sha256=cp_sha,
        calibration_sha256=cal_sha,dataset_sha256=dataset_fingerprint(data),inputs=identity,
        source=source,scripts_sha256=identities,execution=execution,
        environment=dict(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'),
        gc_initial=gc_before,gc_policy_unchanged=True,test_seed=args.test_seed,
        repeats=args.repeats,diagnostic_repeats=args.diagnostic_repeats,rows=rows,scope=SCOPE)


def summarize(workers):
    groups={}
    for w in workers:
        for r in w['rows']:
            groups.setdefault((r['model'],r['retention'],r['record_count']),[]).append((w['training_seed'],r))
    rows=[]
    for key,items in sorted(groups.items()):
        rows.append(dict(model=key[0],retention=key[1],record_count=key[2],checkpoint_pairs=len(items),
            median_checkpoint_ms={m:statistics.median(r['timings'][m]['median_ms'] for _,r in items) for m in MODES},
            median_checkpoint_p90_ms={m:statistics.median(r['timings'][m]['p90_ms'] for _,r in items) for m in MODES},
            per_seed=[dict(training_seed=s,median_ms={m:r['timings'][m]['median_ms'] for m in MODES}) for s,r in items]))
    return rows


def parent(args):
    if args.output_dir.exists():raise ValueError('refusing existing output folder')
    jobs=[]
    for cp in sorted(args.replication_dir.glob('weights-*.pt')):
        seed=int(cp.stem.removeprefix('weights-'));cal=args.replication_dir/f'study-{seed}/calibration.json'
        if not cal.is_file():raise ValueError('missing calibration')
        jobs.append((seed,cp,cal,digest(cp),digest(cal)))
    if not jobs or len({x[3] for x in jobs})!=len(jobs):raise ValueError('distinct checkpoints required')
    random.Random(args.order_seed).shuffle(jobs)
    args.output_dir.mkdir(parents=True)
    source,identities=original.source_identity(),scripts_identity()
    write_json(args.output_dir/'PLAN.json',dict(format=FORMAT+'.plan',scope=SCOPE,source=source,
        scripts_sha256=identities,test_seed=args.test_seed,record_counts=args.record_counts,models=args.models,
        repeats=args.repeats,diagnostic_repeats=args.diagnostic_repeats,warmup=args.warmup,
        order_seed=args.order_seed,job_order=[j[0] for j in jobs],cpu_threads=args.cpu_threads))
    workers=[]
    for seed,cp,cal,ch,ah in jobs:
        path=args.output_dir/f'seed-{seed}.json'
        cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp.resolve()),
            '--calibration',str(cal.resolve()),'--worker-output',str(path.resolve()),'--device',args.device,
            '--record-counts',*map(str,args.record_counts),'--models',*args.models]
        for k in ('test_seed','repeats','diagnostic_repeats','warmup','order_seed','cpu_threads'):
            cmd+=['--'+k.replace('_','-'),str(getattr(args,k))]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w=read_json(path)
        if (w['training_seed']!=seed or w['checkpoint_sha256']!=ch or w['calibration_sha256']!=ah
                or digest(cp)!=ch or digest(cal)!=ah or w['source']!=source or w['scripts_sha256']!=identities):
            raise ValueError('worker identity mismatch')
        if workers and (w['dataset_sha256'],w['inputs'])!=(workers[0]['dataset_sha256'],workers[0]['inputs']):
            raise ValueError('cohorts differ')
        workers.append(w)
    result=dict(format=FORMAT+'.complete',checkpoint_runs=len(workers),scope=SCOPE,
        all_outputs_and_bytes_agree=True,all_gc_policies_unchanged=True,
        rows=summarize(workers),worker_reports=[f'seed-{w["training_seed"]}.json' for w in workers])
    write_json(args.output_dir/'summary.json',result)
    print('\nmodel / receipt / count:     per-record       bulk    chunk16    chunk64')
    for r in result['rows']:
        print(f'{r["model"]}/{r["retention"]}/{r["record_count"]}: '+
              ' '.join(f'{r["median_checkpoint_ms"][m]:10.3f}' for m in MODES))
    print('COMPLETE:',args.output_dir/'summary.json')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replication-dir','output-dir','checkpoint','calibration','worker-output'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--models',nargs='+',choices=['shared','fixed_depth_4','direct'],default=['shared','fixed_depth_4'])
    p.add_argument('--record-counts',nargs='+',type=positive,default=[64,256])
    p.add_argument('--repeats',type=positive,default=30)
    p.add_argument('--diagnostic-repeats',type=original.nonnegative,default=10)
    p.add_argument('--warmup',type=original.nonnegative,default=2)
    p.add_argument('--cpu-threads',type=positive,default=1)
    p.add_argument('--test-seed',type=original.nonnegative,default=46000)
    p.add_argument('--order-seed',type=original.nonnegative,default=49000)
    args=p.parse_args(argv)
    try:
        if (len(set(args.record_counts))!=len(args.record_counts) or max(args.record_counts)>4096
                or len(set(args.models))!=len(args.models)):
            raise ValueError('counts/models must be unique, counts at most 4096')
        if args.worker_output is not None:
            if args.checkpoint is None or args.calibration is None:raise ValueError('worker needs reference files')
            if args.worker_output.exists():raise ValueError('refusing existing worker report')
            write_json(args.worker_output,worker(args))
        else:
            if args.replication_dir is None or args.output_dir is None:raise ValueError('supply replication/output directory')
            parent(args)
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,subprocess.CalledProcessError) as e:
        print('Error:',e,file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
