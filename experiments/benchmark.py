"""Exploratory CPU microbenchmark; not a scientific speedup claim.

All three paths emit a final witness. Full/compact modes also log steps.
Setup and all-candidate routing are measured separately and included in the
reported per-trial total. Warmup excludes interpreter/import startup.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from universa_recurrent.examples import flow_example
from universa_recurrent.recurrence import solve, direct_solve
from universa_recurrent.routing import select_constraint
from universa_recurrent.verification import check_optimality, verify_record


def elapsed(start):
    return (perf_counter_ns()-start)/1e6


def trial(seed: int, method: str) -> dict:
    start=perf_counter_ns()
    example=flow_example(seed)
    setup_ms=elapsed(start)
    start=perf_counter_ns()
    route=select_constraint(example.candidates,example.problem,
                            example.validation_measurement,example.validation_observed)
    routing_ms=elapsed(start)
    if route.chosen is None:
        return dict(seed=seed,method=method,status="refused",setup_ms=setup_ms,routing_ms=routing_ms)
    c,p=route.chosen,example.problem
    start=perf_counter_ns()
    if method == 'direct':
        z=direct_solve(p,c)
        dual=-c.multiplier_map @ p.gradient(z)
        # A final-only certificate record, deliberately not the recurrence schema.
        record={"schema":"benchmark.direct_final.v1", "problem":{
            "measurement":p.measurement.tolist(),"observed":p.observed.tolist(),
            "boundary":c.boundary.tolist(),"ridge":p.ridge},
            "final":{"state":z.tolist(),"multiplier":dual.tolist()}}
        iterations=0
    else:
        out=solve(p,c,trace_mode=method)
        record=out.record
        z=out.state
        iterations=out.iterations
    solve_and_record_ms=elapsed(start)
    start=perf_counter_ns()
    encoded=json.dumps(record,sort_keys=True,allow_nan=False).encode('utf-8')
    serialization_ms=elapsed(start)
    start=perf_counter_ns()
    # Check deserialized evidence, not unrecorded producer arrays.
    payload=json.loads(encoded)
    if method == 'direct':
        pp,ff=payload['problem'],payload['final']
        check=check_optimality(pp['measurement'],pp['observed'],pp['boundary'],
                              ff['state'],ff['multiplier'],ridge=pp['ridge'])
    else:
        check=verify_record(payload)
    verification_ms=elapsed(start)
    return dict(seed=seed,method=method,status="checked" if check.accepted else "failed",
                setup_ms=setup_ms,routing_ms=routing_ms,
                solve_and_record_ms=solve_and_record_ms,serialization_ms=serialization_ms,
                verification_ms=verification_ms,
                total_ms=setup_ms+routing_ms+solve_and_record_ms+serialization_ms+verification_ms,
                bytes=len(encoded),iterations=iterations,
                objective=p.loss(z),feasibility=check.feasibility,stationarity=check.stationarity)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats < 1 or args.output.exists():
        parser.error('repeats must be positive and output must not already exist')
    for method in ('direct','compact','full'):
        trial(7,method)
    rows=[]
    for repeat in range(args.repeats):
        # Rotate order to reduce an obvious fixed-order timing bias.
        methods=['direct','compact','full']
        methods=methods[repeat%3:]+methods[:repeat%3]
        for seed in (7,8,9):
            for method in methods:
                rows.append(dict(repetition=repeat,**trial(seed,method)))
    result={"status":"exploratory_microbenchmark_not_preregistered",
            "scope":"fixed toy graph; three public development seeds; dense CPU arrays",
            "excluded":"interpreter/import startup, package installation, disk writing",
            "warmup":"one full pipeline per method before measurements",
            "metadata":{"python":platform.python_version(),"numpy":np.__version__,
                        "platform":platform.platform(),
                        "thread_environment":{k:os.environ.get(k) for k in
                            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
            "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('Exploratory toy timings; no inference or ML speedup claim.')
    for method in ('direct','compact','full'):
        ok=[r for r in rows if r['method']==method and r['status']=='checked']
        if not ok:
            print(method, 'NO CHECKED TRIALS')
        else:
            print(f"{method:8s} median total {np.median([r['total_ms'] for r in ok]):.3f} ms; "
                  f"median record {np.median([r['bytes'] for r in ok]):.0f} bytes")
    print(f'Raw rows: {args.output}')
    return 0 if all(r['status']=='checked' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
