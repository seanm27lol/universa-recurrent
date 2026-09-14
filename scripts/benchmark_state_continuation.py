"""Pause a frozen solver, encode its working state, reconstruct, and continue.

This is a numerical sufficiency experiment, not learned language interpretation.
Every representation resumes at ONE cut for the SAME remaining update budget.
Original observations, masks, encoder context and prior logits remain available
ONLY to the solver. The codec cannot read them. No future state goes into a codec.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from continuation_codec import StateCodec, BITS, describe
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import (load_estimators, read_json, write_json,
    dataset_fingerprint, source_identity)
from universa_recurrent.neural.retention_study import configure, compare_outputs, ATOL, RTOL
from universa_recurrent.neural.train import sha256_file
from universa_recurrent.neural.v2 import MultiHypothesisRecurrentNet, UntiedMultiHypothesisNet

FORMAT = 'universa-recurrent.state-continuation.v1'
MODELS = ('shared', 'fixed_depth_4')
SCOPE = {
    'question': 'Can a smaller numerical description preserve continuation at a specified cut?',
    'dynamic_state': 'All candidate cycle coordinates AND current route logits.',
    'retained_context': 'Original observations, masks, encoder context and prior logits remain with the solver and are counted separately; the decoder gets none of them.',
    'decoder': 'Only one encoded state and a pinned codec manifest; no future states or answer labels.',
    'intervention': 'Replace the dynamic state once, then execute the original remaining updates.',
    'controls': 'Raw float32 roundtrip, equal-payload rotated quantization, zero state, other-input state, separate coordinate/logit erasure, and stopping at the original cut.',
    'budget': 'Actual value bytes, per-record codec hash, shared codec metadata, and retained-context tensor bytes reported separately. Not total model compression.',
    'meaning': 'Named mathematical fields have known meanings by construction. This is not a trained NLA, a free-text decoder, general latent semantics, or a causal abstraction proof.',
    'evidence': 'Continuation similarity does not alone show that a description is meaningful. Zero/erasure controls test whether original context simply allows recovery.',
    'timing': 'One unoptimized descriptive encode/decode/continuation timing per case, not a latency benchmark; no speedup selection is performed.',
    'sampling': 'Same calibration and test cohorts for all checkpoints. Cuts and bit widths fixed before this run. Different models retain their original differing training objectives.',
}


class NumericalContinuationError(ValueError):
    """A finite reconstruction produced nonfinite continuation: a reported failure."""


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def natural(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def cuts_for(steps):
    if type(steps) is not int or steps < 2:
        raise ValueError('a continuation experiment requires at least two steps')
    return sorted({max(1, steps // 4), max(1, steps // 2), steps - 1})


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def source():
    root = Path(__file__).resolve().parent
    return dict(package=source_identity(), scripts={name: sha256_file(root/name) for name in
                ('benchmark_state_continuation.py', 'continuation_codec.py')})


@dataclass(frozen=True)
class CutState:
    """Explicit continuation boundary; tensors are never modified in-place here."""
    observed: torch.Tensor
    mask: torch.Tensor
    context: torch.Tensor
    prior_logits: torch.Tensor
    coordinates: torch.Tensor
    current_logits: torch.Tensor
    cut: int
    model_identity: int

    def vector(self):
        return torch.cat((self.coordinates.flatten(1), self.current_logits), -1).detach().cpu().numpy().copy()

    def retained_bytes_per_example(self):
        return sum(t[0].numel() * t.element_size() for t in
                   (self.observed, self.mask, self.context, self.prior_logits))


def _step(model, depth, state, coords, logits):
    args = (state.context, state.observed, state.mask, coords, state.prior_logits, logits)
    if isinstance(model, UntiedMultiHypothesisNet):
        return model._step_at_depth(depth, *args)
    return model._step(*args)


@torch.inference_mode()
def pause(model, observed, mask, cut):
    if not isinstance(model, MultiHypothesisRecurrentNet) or model.training:
        raise ValueError('evaluated recurrent v2 model required')
    if type(cut) is not int or not 1 <= cut < model.config.steps:
        raise ValueError('cut must leave at least one original update')
    model._validate_inputs(observed, mask, validate_values=True)
    if len(observed) < 1:
        raise ValueError('empty batch')
    context = model._context(observed, mask)
    prior = model.prior_router(context)
    coords = observed.new_zeros(len(observed), model.config.num_structures, model.config.latent_dim)
    current = prior
    snapshot = CutState(observed, mask, context, prior, coords, current, cut, id(model))
    for depth in range(cut):
        coords, _, current, _, _ = _step(model, depth, snapshot, coords, current)
    return CutState(observed, mask, context, prior, coords, current, cut, id(model))


@torch.inference_mode()
def continue_from(model, state, restored, policy, *, stop_at_cut=False):
    if model.training or state.model_identity != id(model):
        raise ValueError('continuation must use the same evaluated model instance')
    if type(state.cut) is not int or not 1 <= state.cut < model.config.steps:
        raise ValueError('invalid continuation cut')
    k, d = model.config.num_structures, model.config.latent_dim
    vector = np.asarray(restored)
    if (vector.shape != (len(state.observed), k * (d + 1)) or vector.dtype != np.float32
            or not np.isfinite(vector).all()):
        raise ValueError('restored state must be finite float32 with all coordinates and logits')
    # Clone the decoded working vector. No original dynamic values are reused.
    working = torch.from_numpy(vector.copy()).to(state.observed.device)
    coords = working[:, :k*d].reshape(-1, k, d)
    logits = working[:, k*d:]
    executed = 0
    if not stop_at_cut:
        for depth in range(state.cut, model.config.steps):
            coords, _, logits, _, _ = _step(model, depth, state, coords, logits)
            executed += 1
    states = torch.einsum('knd,bkd->bkn', model.bases, coords)
    if not torch.isfinite(states).all() or not torch.isfinite(logits).all():
        raise NumericalContinuationError('nonfinite continued candidate state or route evidence')
    output = construct_output(states, logits.softmax(-1), policy, 'mixture_with_claim')
    return output, executed


def as_cpu(output):
    return {key: value.detach().cpu() for key, value in output.items()}


def effects(out, reference, truth, labels, original_state, restored_state):
    out, reference = as_cpu(out), as_cpu(reference)
    a, b = out['estimate'].double(), reference['estimate'].double()
    task_error = (a - truth.double()).square().mean(-1)
    reference_error = (b - truth.double()).square().mean(-1)
    delta = (a - b).square().mean(-1)
    accepted = out['claim_mask']
    wrong = accepted & (out['top_route'] != labels)
    state_delta = (np.asarray(original_state, dtype=np.float64) - np.asarray(restored_state, dtype=np.float64))**2
    near = torch.isclose(a, b, atol=ATOL, rtol=RTOL).all(-1)
    per_example = dict(final_estimate_mse_to_reference=delta.numpy(), task_mse=task_error.numpy(),
        reference_task_mse=reference_error.numpy(), state_mse=state_delta.mean(-1),
        top_route_changed=(out['top_route'] != reference['top_route']).numpy(),
        claim_changed=(out['claim_index'] != reference['claim_index']).numpy(),
        estimate_within_tolerance=near.numpy(), claimed=accepted.numpy(), wrong_claim=wrong.numpy())
    return dict(n=len(a), final_estimate_mse_to_reference=float(delta.mean()),
        max_absolute_estimate_change=float((a-b).abs().max()),
        task_mse=float(task_error.mean()), reference_task_mse=float(reference_error.mean()),
        task_mse_change=float((task_error-reference_error).mean()),
        state_mse=float(state_delta.mean()),
        probability_max_absolute_change=float((out['probabilities']-reference['probabilities']).abs().max()),
        top_route_changed_n=int((out['top_route'] != reference['top_route']).sum()),
        claim_mask_changed_n=int((accepted != reference['claim_mask']).sum()),
        claim_index_changed_n=int((out['claim_index'] != reference['claim_index']).sum()),
        estimate_within_tolerance_n=int(near.sum()), claimed_n=int(accepted.sum()),
        wrong_claim_n=int(wrong.sum()),
        wrong_claim_rate_claimed=int(wrong.sum())/int(accepted.sum()) if accepted.any() else None), per_example


def state_cases(calibration, original, site, bits, rotation_seed):
    raw = StateCodec.fit(calibration, kind='raw_f32', site=site)
    yield 'raw_f32', raw, original, False
    for b in bits:
        for kind in ('named', 'rotated'):
            codec = StateCodec.fit(calibration, kind=kind, bits=b, site=site, rotation_seed=rotation_seed)
            yield f'{kind}_{b}bit', codec, original, False
    # Controls are interventions, not deployable codecs or independent test data.
    yield 'zero_dynamic', None, np.zeros_like(original), False
    yield 'other_input_dynamic', None, np.roll(original, 1, axis=0).copy(), False
    ncoords = site['num_structures'] * site['latent_dim']
    no_coords = original.copy(); no_coords[:, :ncoords] = 0
    no_logits = original.copy(); no_logits[:, ncoords:] = 0
    yield 'erase_coordinates', None, no_coords, False
    yield 'erase_current_logits', None, no_logits, False
    yield 'stop_at_cut', None, original, True


@torch.inference_mode()
def run_worker(args):
    if args.calibration_seed == args.test_seed or args.n < 2:
        raise ValueError('disjoint calibration/test seeds and at least two test inputs required')
    torch.set_num_threads(1)
    execution = configure(True)
    start_identity = source()
    cp_sha, policy_sha = sha256_file(args.checkpoint), sha256_file(args.claim_calibration)
    engines, metadata, device, reserved = load_estimators(args.checkpoint, args.device)
    policy_artifact = read_json(args.claim_calibration)
    if metadata['checkpoint_sha256'] != cp_sha or policy_artifact.get('checkpoint_sha256') != cp_sha:
        raise ValueError('checkpoint and claim-calibration identity mismatch')
    forbidden = reserved | {policy_artifact['data']['seed']}
    if 'test_seed_reserved' in policy_artifact:
        forbidden.add(policy_artifact['test_seed_reserved'])
    if {args.calibration_seed, args.test_seed} & forbidden:
        raise ValueError('codec calibration/test split overlaps previous training or claim calibration')
    settings = metadata['training']
    kwargs = {key: settings[key] for key in ('noise_std', 'observe_probability')}
    calibration = StructuredFlowDataset(args.calibration_size, seed=args.calibration_seed, **kwargs)
    test = StructuredFlowDataset(args.n, seed=args.test_seed, **kwargs)
    uniqueness = len({row.tobytes() for row in np.concatenate((test.observed.numpy(), test.mask.numpy()), -1)})
    if uniqueness != args.n:
        raise ValueError('test observations/masks must be distinct')
    x, m = test.observed.to(device), test.mask.to(device)
    cx, cm = calibration.observed.to(device), calibration.mask.to(device)
    by_name = {e.name: e for e in engines}
    if not set(args.models) <= set(by_name):
        raise ValueError('missing recurrent controls')
    rows, examples, codecs, arrays, gates = [], [], {}, {}, []
    arrays.update(test_observed=test.observed.numpy(), test_mask=test.mask.numpy(),
                  test_truth=test.truth.numpy(), test_labels=test.label.numpy())
    for name in args.models:
        engine = by_name[name]; model = engine.model.eval()
        if policy_artifact['model_sources'][name] != cp_sha:
            raise ValueError('wrong calibrated model identity')
        policy = ClaimPolicy(**policy_artifact['models'][name]['policy'])
        # The original rollout is the comparison, not a newly written implementation.
        h = model.rollout(x, m)
        reference = construct_output(h['states'][-1], h['route_probabilities'][-1], policy, 'mixture_with_claim')
        reference = as_cpu(reference)
        del h
        for cut in cuts_for(model.config.steps):
            point = pause(model, x, m, cut)
            cal_point = pause(model, cx, cm, cut)
            original, cal = point.vector(), cal_point.vector()
            del cal_point
            arrays[f'{name}__cut{cut}__original_dynamic'] = original.copy()
            arrays[f'{name}__reference_estimate'] = reference['estimate'].numpy()
            arrays[f'{name}__reference_claim_index'] = reference['claim_index'].numpy()
            site = dict(checkpoint_sha256=cp_sha, model=name, cut=cut, depth=model.config.steps,
                        num_structures=model.config.num_structures, latent_dim=model.config.latent_dim,
                        names=metadata['names'], calibration_sha256=dataset_fingerprint(calibration))
            full, steps = continue_from(model, point, original, policy)
            gate = compare_outputs(full, reference)
            if steps != model.config.steps-cut:
                raise ValueError('wrong remaining update budget')
            gates.append(dict(model=name, cut=cut, full_state_restoration=gate, steps=steps))
            print(f'{name}: exact-state gate passed at cut {cut}/{model.config.steps}', flush=True)
            for method, codec, value, stop in state_cases(cal, original, site, args.bits, args.rotation_seed):
                key = f'{name}__cut{cut}__{method}'
                t = time.perf_counter_ns()
                payloads, clipping = (codec.encode(value) if codec is not None else (None, None))
                encoded_ms = (time.perf_counter_ns()-t)/1e6
                t = time.perf_counter_ns()
                restored = codec.decode(payloads) if codec is not None else value.copy()
                decoded_ms = (time.perf_counter_ns()-t)/1e6
                if codec is not None:
                    codecs[key] = codec.manifest()
                sync(device); t = time.perf_counter_ns()
                try:
                    output, steps = continue_from(model, point, restored, policy, stop_at_cut=stop)
                except NumericalContinuationError as error:
                    if method == 'raw_f32':
                        raise
                    rows.append(dict(model=name, cut=cut, depth=model.config.steps, method=method,
                        status='nonfinite_continuation', failure=str(error), metrics=None, clipping=clipping,
                        storage=None if codec is None else codec.storage(args.n)))
                    print(f'  {method}: NONFINITE CONTINUATION (retained as a failure)', flush=True)
                    continue
                sync(device); resumed_ms = (time.perf_counter_ns()-t)/1e6
                if not stop and steps != model.config.steps-cut:
                    raise ValueError('codec changed the remaining budget')
                if method == 'raw_f32':
                    if not np.array_equal(restored, original):
                        raise ValueError('raw float32 state roundtrip changed data')
                    compare_outputs(output, reference)
                measurements, values = effects(output, reference, test.truth, test.label, original, restored)
                ncoords = model.config.num_structures * model.config.latent_dim
                measurements['coordinate_mse'] = float(np.square(original[:, :ncoords].astype(float)-restored[:, :ncoords]).mean())
                measurements['route_logit_mse'] = float(np.square(original[:, ncoords:].astype(float)-restored[:, ncoords:]).mean())
                rows.append(dict(model=name, cut=cut, depth=model.config.steps, method=method, status='finite',
                    original_remaining_updates=model.config.steps-cut, executed_updates=steps,
                    metrics=measurements, clipping=clipping,
                    storage=None if codec is None else codec.storage(args.n),
                    retained_context_bytes_per_example=point.retained_bytes_per_example(),
                    original_dynamic_bytes_per_example=int(original.shape[1]*4),
                    single_pass_cost_ms=dict(encode=encoded_ms, decode=decoded_ms,
                                             transfer_restore_resume_and_output=resumed_ms)))
                values.update(restored_dynamic=restored.copy(), final_estimate=output['estimate'].detach().cpu().numpy(),
                              final_probabilities=output['probabilities'].detach().cpu().numpy(),
                              final_claim_index=output['claim_index'].detach().cpu().numpy())
                for field, array in values.items():
                    arrays[f'{key}__{field}'] = array
                # Save first two in order, not cherry-picked, for readable inspection.
                if codec is not None and codec.manifest()['kind'] != 'rotated':
                    for i in range(min(2, args.n)):
                        text = describe(restored[i], names=metadata['names'], latent_dim=model.config.latent_dim, cut=cut)
                        examples.append(dict(key=key, input_index=i, description=text,
                            description_utf8_bytes=len(text.encode()),
                            payload_hex=payloads[i].hex(), codec_identity=codec.identity.hex(),
                            original_dynamic=original[i].tolist(), restored_dynamic=restored[i].tolist(),
                            reference_estimate=reference['estimate'][i].tolist(),
                            resumed_estimate=output['estimate'][i].detach().cpu().tolist(),
                            text_role='Display only; decoder consumes the numerical payload, not generated prose.'))
                print(f'  {method:<22} change={measurements["final_estimate_mse_to_reference"]:.3g} '
                      f'claims_changed={measurements["claim_index_changed_n"]}/{args.n}', flush=True)
            # Neither the checkpoint nor the prefix state may have been mutated.
            if not np.array_equal(original, point.vector()):
                raise ValueError('cut state changed in-place')
    if (sha256_file(args.checkpoint) != cp_sha or sha256_file(args.claim_calibration) != policy_sha
            or source() != start_identity):
        raise ValueError('source or reference files changed during study')
    artifact_path = args.output.with_suffix('.npz')
    with artifact_path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return dict(format=FORMAT+'.worker', source=start_identity, execution=execution,
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        training_seed=settings['seed'], checkpoint_sha256=cp_sha, claim_calibration_sha256=policy_sha,
        calibration_seed=args.calibration_seed, test_seed=args.test_seed,
        calibration_sha256=dataset_fingerprint(calibration), test_sha256=dataset_fingerprint(test),
        calibration_n=args.calibration_size, n=args.n, distinct_test_inputs=uniqueness,
        data_settings=kwargs, scope=SCOPE, rows=rows, codecs=codecs, examples=examples,
        restoration_gates=gates, all_restoration_gates_passed=True,
        per_example_file=artifact_path.name, per_example_sha256=sha256_file(artifact_path))


def summarize(workers):
    maps = [{(r['model'], r['cut'], r['method']): r for r in w['rows']} for w in workers]
    keys = set(maps[0])
    if any(set(m) != keys for m in maps):
        raise ValueError('worker cases differ')
    def spread(values):
        if any(v is None for v in values):
            return dict(mean=None, sd=None, per_checkpoint=values, note='At least one failed continuation; no aggregate ranking.')
        return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else None,
                    per_checkpoint=values)
    return [dict(model=k[0], cut=k[1], method=k[2],
                 failed_checkpoint_n=sum(m[k]['metrics'] is None for m in maps),
                 final_change=spread([m[k]['metrics']['final_estimate_mse_to_reference'] if m[k]['metrics'] is not None else None for m in maps]),
                 task_mse=spread([m[k]['metrics']['task_mse'] if m[k]['metrics'] is not None else None for m in maps]),
                 claim_change_fraction=spread([m[k]['metrics']['claim_index_changed_n']/m[k]['metrics']['n'] if m[k]['metrics'] is not None else None for m in maps]))
            for k in sorted(keys)]


def parent(args):
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite output directory')
    jobs = []
    for cp in sorted(args.replication_dir.glob('weights-*.pt')):
        seed = int(cp.stem.removeprefix('weights-'))
        policy = args.replication_dir/f'study-{seed}/calibration.json'
        if not policy.is_file():
            raise ValueError(f'missing claim calibration for {seed}')
        jobs.append((seed, cp, policy, sha256_file(cp), sha256_file(policy)))
    if not jobs or len(set(j[3] for j in jobs)) != len(jobs):
        raise ValueError('distinct checkpoints required')
    random.Random(args.rotation_seed).shuffle(jobs)
    initial_source = source()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'PLAN.json', dict(format=FORMAT+'.plan', scope=SCOPE, source=initial_source,
        models=args.models, bits=args.bits, calibration_seed=args.calibration_seed,
        test_seed=args.test_seed, n=args.n, calibration_size=args.calibration_size,
        rotation_seed=args.rotation_seed, job_order=[j[0] for j in jobs], no_training=True))
    workers = []
    for seed, cp, cal, ch, ah in jobs:
        output = args.output_dir/f'seed-{seed}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--checkpoint', str(cp.resolve()),
            '--claim-calibration', str(cal.resolve()), '--output', str(output.resolve()),
            '--device', args.device, '--models', *args.models, '--bits', *map(str, args.bits)]
        for name in ('n', 'calibration_size', 'calibration_seed', 'test_seed', 'rotation_seed'):
            command.extend(['--'+name.replace('_','-'), str(getattr(args, name))])
        subprocess.run(command, check=True, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w = read_json(output)
        if (w['training_seed'] != seed or w['checkpoint_sha256'] != ch or w['claim_calibration_sha256'] != ah
                or sha256_file(cp) != ch or sha256_file(cal) != ah or w['source'] != initial_source):
            raise ValueError('worker reference/source mismatch')
        if workers and (w['test_sha256'], w['calibration_sha256']) != (workers[0]['test_sha256'], workers[0]['calibration_sha256']):
            raise ValueError('different worker cohorts')
        workers.append(w)
    result = dict(format=FORMAT+'.complete', checkpoint_runs=len(workers), scope=SCOPE,
        worker_files=[f'seed-{w["training_seed"]}.json' for w in workers],
        source=initial_source, test_sha256=workers[0]['test_sha256'], calibration_sha256=workers[0]['calibration_sha256'],
        all_restoration_gates_passed=all(w['all_restoration_gates_passed'] for w in workers),
        rows=summarize(workers), warning='Descriptive checkpoint spread on one shared test cohort; no semantic or generalization claim.')
    write_json(args.output_dir/'summary.json', result)
    print('COMPLETE:', args.output_dir/'summary.json', flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('replication-dir','output-dir','checkpoint','claim-calibration','output'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    p.add_argument('--bits', type=int, choices=BITS, nargs='+', default=list(BITS))
    p.add_argument('--calibration-size', type=positive, default=1024)
    p.add_argument('--n', type=positive, default=1024)
    p.add_argument('--calibration-seed', type=natural, default=61000)
    p.add_argument('--test-seed', type=natural, default=62000)
    p.add_argument('--rotation-seed', type=natural, default=63000)
    args = p.parse_args(argv)
    try:
        if (len(set(args.models)) != len(args.models) or len(set(args.bits)) != len(args.bits)
                or args.calibration_seed == args.test_seed or args.n < 2):
            raise ValueError('unique cases, disjoint seeds and at least two test examples required')
        if args.output is not None:
            if args.checkpoint is None or args.claim_calibration is None:
                raise ValueError('worker requires checkpoint and claim calibration')
            if args.output.exists() or args.output.with_suffix('.npz').exists():
                raise ValueError('refusing to overwrite worker artifacts')
            write_json(args.output, run_worker(args))
        else:
            if args.replication_dir is None or args.output_dir is None:
                raise ValueError('supply replication-dir and output-dir')
            parent(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, FloatingPointError, subprocess.CalledProcessError) as error:
        print('Error:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
