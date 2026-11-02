# Independent verifiers (the reviewer's)

Written by the independent reviewer, apart from the producer modules. They read the raw release, the pins and
`contracts.py` *at each run's commit* (`git show`), `sqlite3`, and their own EX rule. They are not producer code;
no run imports them.

    nice -n 19 python3 verify_baselines.py RUN_DIR [RUN_DIR ...] [--repo REPO] [--runs-root RUNS] [--no-rehash] [--json OUT]
    nice -n 19 python3 verify_smoke.py SMOKE_RUN_DIR [--repo REPO] [--runs-root RUNS] [--no-rehash] [--redecode]
    nice -n 19 python3 verify_pilot.py [--representability] [--redecode] [--bundle research/ti_tnn/scaling/evidence/pilot_v1] [--json OUT]

`verify_pilot.py` reconstructs the C4/T4 pilot's verdict from its records. It re-executes every final SQL with its
own EX rule, and recomputes the counts, gates, differences, rule outcome, database table and sign-flip analysis.
It also re-derives each checkpoint's digests and the training stream, and parses the ledger. Its output on the pilot
is in `../evidence/pilot_v1/independent/`. Its `--repo` default was changed from an absolute path to this checkout
when it was copied here.

`verify_smoke.py --redecode` is the one part that uses producer code (`scaling.encode`, `infer_controller`, `train`),
for decoding only. It re-decodes the 16 development requests on CPU from the final checkpoints and the cached GPU
states, then compares every action and the final SQL with the records.

They were copied here from the review's scratch directory with two path-only changes: the gold-execution cache
moved to `$TI_TNN_VERIFY_CACHE` (default `~/.cache/ti_tnn_scaling_verify`), and `verify_smoke.py` finds `research/ti_tnn`
relative to itself. Their output on the development runs is in `../evidence/dev_v1/independent_verification.txt`.
