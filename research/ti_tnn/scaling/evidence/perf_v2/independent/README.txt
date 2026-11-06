Independent verification of the performance study v2 (outputs copied by the producer session; the verifier and
its runs are the reviewer's).

verify_perf.json: the final run of independent/verify_perf.py (sha256 d0292a584600faf0..., run 2026-11-03T13:54Z
at checkout 4ff35e1 against the runs root): 813 checks, 0 failures.
  pins 37, history 36, ledger 16, baselines 266, train 349, eval 48, compare 61.
  compare: every one of analysis.json's 3,143 leaves equals the reviewer's re-derivation.
  Producer code used: only the gold-program strata classification (perf_analysis.strata), in a separate process
  from the frozen worktree; everything else (digests, sampling stream, EX rule, rules) is the reviewer's own.
  Warnings (no score changes): two submitted queries scored incorrect in 13.9 s and 15.9 s during evaluation time
  out at 30 s on the reviewer's machine load (s100 C4-v1 bird23tf-00472, s101 C4-v1 bird23tf-03149); records
  spell the shared arm C4_params, configs c4_params.
  Not checked independently: the strata classification, per-request MAC counts, the shared trunk initialization
  at update 0 (no update-0 checkpoint), the loss values (nothing retrained), CPU re-decoding, the U = 3000
  feasibility decision.

verify_perf_phase1.json: phase 1 (pins, history, ledger, baselines) before training, 2026-11-02: 221 checks,
0 failures. Kept as recorded.
