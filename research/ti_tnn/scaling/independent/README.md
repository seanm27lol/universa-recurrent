# Independent verifiers (the reviewer's)

Written by the independent reviewer, apart from the producer modules. They read the raw release, the pins and
`contracts.py` *at each run's commit* (`git show`), `sqlite3`, and their own EX rule. They are not producer code;
no run imports them.

    nice -n 19 python3 verify_baselines.py RUN_DIR [RUN_DIR ...] [--repo REPO] [--runs-root RUNS] [--no-rehash] [--json OUT]
    nice -n 19 python3 verify_smoke.py SMOKE_RUN_DIR [--repo REPO] [--runs-root RUNS] [--no-rehash] [--redecode]
    nice -n 19 python3 verify_pilot.py [--bundle DIR] [--json OUT] [--local [--runs-root RUNS] [--representability] [--redecode]]

`verify_pilot.py` reconstructs the C4/T4 pilot's verdict in two parts, reported separately:
- **The public arithmetic reconstruction** (always) reads only the evidence bundle, plus the checkout's pins,
  protocol and git history:
  - each evaluation's eligibility, from its complete reconstruction inventory;
  - the producer's totals and aggregate flags, recomputed from the records;
  - the record bindings;
  - the counts, gates, differences, rule, histogram, database table and sign-flip, from the recorded EX scores;
  - the training records;
  - the ledger;
  - agreement with `analysis.json`, with runs matched by their path relative to the runs root.
- **The local checks** (`--local`) need the runs root:
  - the 64, rebuilt from the raw release;
  - every final SQL, re-executed with this directory's own EX rule;
  - the final checkpoints' hashes and digests;
  - the bundle's copies, compared with their run originals.

**History of `verify_pilot.py`:**
- **Written by the reviewer.** Its `--repo` default was changed from an absolute path to this checkout when it was
  copied here.
- **The reviewer's outputs:** `pilot_verdict.json`, `pilot_verdict_redecode.json`, `pilot_verdict_summary.json` and
  `verify_L64.json` in `../evidence/pilot_v1/independent/` came from the version committed in 0c420ab, run against an
  earlier 84-file snapshot of the bundle. They are preserved unchanged and labelled there.
- **Changed on 2026-11-02, at the user's request, by the producer-side session, not by the reviewer:**
  - the public/local split;
  - eligibility derived from the inventory, so a false entry rejects a run whose summary still says `passed`;
  - the recomputed totals and serving accounting;
  - the run references.
  Its output on the final bundle is `../evidence/pilot_v1/independent/final_bundle_verification.json`.

`verify_smoke.py --redecode` is the one part that uses producer code (`scaling.encode`, `infer_controller`, `train`),
for decoding only. It re-decodes the 16 development requests on CPU from the final checkpoints and the cached GPU
states, then compares every action and the final SQL with the records.

They were copied here from the review's scratch directory with two path-only changes: the gold-execution cache
moved to `$TI_TNN_VERIFY_CACHE` (default `~/.cache/ti_tnn_scaling_verify`), and `verify_smoke.py` finds `research/ti_tnn`
relative to itself. Their output on the development runs is in `../evidence/dev_v1/independent_verification.txt`.
