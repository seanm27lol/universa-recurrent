The second independent verification (the correction pass of 2026-11-07), copied by the producer session; the
verifier and its run are the reviewer's. The first run (167 checks, 1 failure) is one directory up, unchanged.

independent/verify_spr.py, sha256 daa612c3d6ba2324..., recorded by the running process from its own bytes at start
and end. The run: 2026-11-07T03:08:16Z to 03:14:21Z (364.7 s), at d0eec03. It hashed 117 inputs, and its scan of
6,465 files is recorded as one digest. Result: 191 checks, 0 failures.

The historical defect, the original analysis.json's rule fields at 7 of 128, is recorded under `known_defects` exactly
as first found. It is not counted as a failure and not suppressed. The corrected analysis (spr-rule-v2) uses +8 in
every rule field, matches the reviewer's own computation, and leaves the counts and the decision equal to the
original's. score_mapping.jsonl and preservation_own.jsonl of this run are byte-identical to the first run's, so they
are not copied again.
