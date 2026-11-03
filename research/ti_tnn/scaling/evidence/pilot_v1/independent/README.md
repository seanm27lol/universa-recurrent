# Verification outputs in this bundle

Two verifications are kept here. They ran against different states of the bundle, and they are labelled so that
neither is read as describing the other.

## The reviewer's original outputs (2026-11-02, preserved byte for byte)

| file | what it is |
|---|---|
| `pilot_verdict.json` | the reviewer's verdict: 133 checks, 0 failed |
| `pilot_verdict_redecode.json` | the same, with the CPU re-decode (informational): 130 checks, 0 failed |
| `pilot_verdict_summary.json` | the reviewer's summary of the two |
| `verify_L64.json` | `verify_baselines.py` on the L run: 1,348 checks, 0 failed |

They came from the reviewer's `independent/verify_pilot.py` as committed in 0c420ab, and from `verify_baselines.py`.

**They describe an earlier snapshot of this bundle.** When they ran, the bundle held 84 files: the run copies, without
`quarantine/20261102T015059Z/quarantine.json` and without any of these outputs. Their `bundle` sections ("files": 84,
with the quarantine record listed as not bundled) describe that snapshot, not this bundle. The quarantine record was
added afterwards; it was the reviewer's first discrepancy.

**The 84 run copies are unchanged since.** Each equalled its run original then, and each equals it now (the local
checks in `final_bundle_verification.json`).

**Two limits of that version**, both corrected in the final-bundle verification below:
- an evaluation's eligibility required the producer's top-level `passed` flag but did not derive it from the complete
  reconstruction inventory;
- `analysis.json`'s inputs were compared by absolute path.

Neither affects its verdict: both inventories are entirely true, and the absolute paths were equal.

## The verification of the final bundle (`final_bundle_verification.json`)

**Who ran it, and with what.** The producer-side session ran the corrected `independent/verify_pilot.py` at the
user's request; it is not a rerun by the reviewer. It verified this bundle as listed by the `SHA256SUMS` whose
sha256 it records (`public_reconstruction.bundle.sha256sums_sha256`). That listing holds every file here except
`final_bundle_verification.json` itself, which was added afterwards, with `SHA256SUMS` rewritten.

It reports two parts separately:
- **`public_reconstruction`** reads only this bundle, plus the checkout's pins, protocol and git history. It derives
  each evaluation's eligibility from its complete reconstruction inventory, and recomputes the producer's totals
  from the records. It then recomputes the counts, gates, differences, rule, histogram, database table and
  sign-flip from the recorded EX scores, and compares them with `analysis.json`, matching runs by their path
  relative to the runs root.
- **`local_checks`** needed the runs root. It rebuilt the 64 from the raw release and re-executed every final SQL
  with the reviewer's EX rule. It also re-hashed the final checkpoints and recomputed their digests, compared the
  bundle with the run originals, recomputed representability (producer grammar, flagged), and re-decoded the 256
  requests on the CPU (producer decoder, flagged, informational).
