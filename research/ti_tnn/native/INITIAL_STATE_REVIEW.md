# Initial-state study: independent CPU review

The terminal multiplication can appear in a read graph while doing nothing.
This review asks whether the new composition selection excludes that case and
whether the fixed eight-run comparison can resume without changing its identity
or spending a fresh allocation. It does not review a learning result.

Two separate review agents inspected the producer's implementation read-only.
The selection checker used its own reference executor, without invoking the
producer's solve or decomposition. These are independent implementation checks
within the same local project, not external replication or additional model seeds.
Private scripts, detailed outputs and final source hashes remain in the owner
handoff; no datasets, checkpoints or operational paths are published here.

## Selection and saved scoring

The reviewer exhaustively evaluated all 128 revised programs over all `32^4`
initial vectors. Every designated multiplication output is nonconstant and every
recorded actual initial vector changes the designated final target under the
declared skip-write ablation. All 128 recorded witnesses match. Exactly 102 v1
examples remain and 26 change. All 263 original artifact hashes remain valid;
233 files match v2 byte-for-byte. The remaining old-file changes are composition
files and their aggregate metadata. Historical/training pattern exclusion passes.
The main contrast, seeds, objective, architecture settings and endpoint are unchanged.

Independent scoring fixtures include a changed write, an unchanged arithmetic
result, malformed output and empty per-register strata. Hand-calculated totals
match: complete answers 1/4, changed registers 1/4, unchanged registers 9/12.
Malformed output contributes two failures to each value category; no-op scores
0/4 changed and 12/12 unchanged. Existing complete-answer scoring stays intact.
The zeroed-multiplication counterexample and both global/actual-state identity
multiplications are rejected. No unresolved selection or scoring findings remain.

## Runner and recovery

Four review findings were resolved before freezing:

| Finding | Resolution |
|---|---|
| Rolling a reservation ID into checkpoint configuration can break an interrupted second recovery | This study uses an immutable owner binding. Attempt IDs stay in logs/results/ledger. Historical trainer behavior is unchanged |
| A final checkpoint may exist before the result JSON is written | Adopt a hash/identity/journal-validated final descriptor without a trainer call. Unjournaled or invalid finals fail closed and are preserved |
| The source binding named a nonexistent dependency-lock location | Bind the actual root lock and verify all installed pins at preparation and preflight; required files cannot be silently omitted |
| A concurrent resume could finish while another resume waited for the lock | Recheck all-arm completion under the comparison lock before creating an attempt or reservation |

The reviewer observed 25 CPU mocked tests pass, including its own evaluation
adapter fixture. The adapter visited all nine populations of 128 prompt-only
records, enforced fixed complete-prompt generation and uncapped CE/readouts,
retained malformed denominators, and preserved a strict-allclose failure alongside
scale-based agreement. No unresolved runner findings remain in the reviewed bytes.

The producer separately ran all **412 native CPU tests**, including historical
reservation/checkpoint/recovery regressions, and the release check. Final-head CI
and the source/protocol binding are linked in the draft PR and private handoff.
No study training, checkpoint inference, real reservation, shaping, WSP or merge
was part of review. Existing tiny CPU test fixtures are engineering tests only.
