# Milestone 2 local handoff — 2026-09-21

Implemented the deterministic five-split plan, the calibration-fitted P4 PCA
baseline, the frozen three-form text editor with coverage accounting,
whole-group bootstrap decision statistics, the calibration/pilot runner
stages, and stage-aware auditing, and repaired the suffix-causality gate after
the first released-model smoke failed. The repaired runner then
**COMPLETED** the eight-group engineering smoke on the released weights with
an independent auditor PASS. This is an engineering handoff plus a real-model
smoke gate, **not** a scientific finding about the released models.

## Actual checks

| Command/check | Actual result | Scope |
|---|---|---|
| `.venv-phase2/bin/python -m pytest -q -c research/open_weight_lingua/pyproject.toml research/open_weight_lingua/tests` | **115 passed in 17.03 s** on the final implementation | Final CPU suite: splits, P4 fit, frozen edit rule, bootstrap statistics, calibration/pilot stages, repaired suffix gates |
| `uvx --from ruff==0.14.14 ruff check research/open_weight_lingua/src research/open_weight_lingua/tests research/open_weight_lingua/scripts` | **PASS** | Static Python checks |
| `bash -n research/open_weight_lingua/scripts/run_calibration.sh research/open_weight_lingua/scripts/run_pilot.sh` | **PASS** | Shell syntax of both new stage scripts |
| `git diff --check` | **PASS** | Whitespace in tracked changes |
| First released-model smoke, main checkout, Milestone 1 code: `bash research/open_weight_lingua/scripts/run_smoke.sh --fetch-models` (run dir `smoke-20260921T185833Z-c264a184`, retained in the main checkout) | Fetch and hash verification of all **41,410,059,546** pinned bytes **PASS**; 32/32 identity gates bitwise **PASS**; 32/32 AV descriptions ok; 32/32 AR reconstructions ok; then **FAILED** in the behavior stage: `RuntimeError: answer suffix changed the causal prefix activation` | Preserved explicitly below; its `completion.json` says `FAILED`. No part of this failure is relabeled as success |
| Suffix probe, local scratch only (`/tmp/owl_suffix_probe/probe.py`, `/tmp/owl_suffix_probe/results.json`; local evidence, **not committed**) | 8/8 same-length different-suffix-content prefix vector pairs bitwise identical on GB10 BF16 **and** CPU fp32; cross-length BF16 drift 2.09e-2–3.09e-2 norm-relative; CPU fp32 collapses to ~4e-6; answer argmax stable 20/20; even fp32 marginally fails the old 1e-5 absolute check | Established that the old gate is unattainable and that the drift is kernel-scheduling noise, not a leak; see the next section |
| Repaired-code smoke rerun, this branch: `bash research/open_weight_lingua/scripts/run_smoke.sh --fetch-models` (run dir `research/open_weight_lingua/runs/smoke-20260921T211151Z-a4e4a038`, kept local) | **COMPLETE**: 8/8 groups successful, 0 failed prompt variants; 336 `suffix_drift_relative` samples, maximum 3.55e-2, inside the frozen 1e-1 bound | First full released-model execution of target/AV/AR on this stack; engineering smoke only, not the pilot |
| `.venv-phase2/bin/python -m open_weight_lingua.audit research/open_weight_lingua/runs/smoke-20260921T211151Z-a4e4a038` | **PASS** (8 groups) | Independent saved-count and KL replay without loading models; does not authenticate execution or regenerate all teacher-forced logits |
| Existing Phase One suite and runtime | **Untouched** | No Phase One source, test, protocol, or sealed-result file is modified on this branch; not re-run for this handoff |

The CPU tests use randomly initialized tiny Qwen2 models and fixed fixture
text; their outputs are never labeled released-Qwen or released-NLA
measurements. The smoke-stage manifest is byte-compatible with the Milestone 1
schema except for the added `suffix_causality_gate`, `suffix_drift_bound`, and
`suffix_drift_bound_justification` fields, and the rerun manifest records
`source_file_sha256` over the repaired files.

## The GB10 suffix-causality finding

Mechanism, established by the retained probe measurements. Qwen's attention
mask enters as a finite-minimum additive bias before a float32 softmax, so
masked positions contribute exactly zero — which is why the same-length
control is bitwise exact and any genuine suffix leak would be caught
deterministically. Across different sequence lengths, cuBLAS re-selects BF16
GEMM kernels with different reduction orders on sm_121, and those
reassociated sums differ at the 1e-2 norm-relative scale, compounding across
the 21 blocks up to the extraction site. The drift collapsed to ~4e-6 in CPU
fp32 and the answer argmax stayed stable in 20/20 appends, so the verdict is
numerical kernel-scheduling noise, not a causal leak. The tolerance-free
same-length dummy-suffix bitwise comparison is the exact discriminator: a
content leak fails it loudly, while kernel noise cannot occur at equal
length.

Repair, on this branch and frozen **before** the pilot (brief §4), not relaxed
after inspecting results:

* `target.py` freezes `SUFFIX_DRIFT_BOUND = 1e-1` with a documented
  justification: measured maximum cross-length relative drift 3.09e-2, safety
  factor about 3, frozen pre-pilot.
* `score_answer` now applies the same-length dummy-suffix bitwise gate
  (RuntimeError on any suffix-content dependence), then the frozen relative
  drift bound, and records per-score `suffix_drift_relative` evidence.
* The manifest gains `suffix_causality_gate`, `suffix_drift_bound`, and
  `suffix_drift_bound_justification` so every run carries the frozen policy.

This finding discriminates a causal leak from kernel-reduction noise only. It
establishes nothing about the semantic content of activations.

## Real-model smoke evidence — engineering data, not scientific results

Run `smoke-20260921T211151Z-a4e4a038`, 8 smoke groups / 32 prompt variants,
all denominators over all variants, no failed or missing cases:

| Condition | Exact answers / 32 | Accuracy | Agreement with P0 | Mean valid next-token KL |
|---|---|---|---|---|
| P0 unmodified | 25 | 0.781 | 1.000 | 0.0 |
| P1 original reinserted (mandatory adapter gate) | 25 | 0.781 | 1.000 | 0.0 |
| P2 own description → AR direction + original norm | 18 | 0.562 | 0.656 | 1.41 |
| P3 another group's description + receiver norm | 15 | 0.469 | 0.531 | 4.63 |
| P5 norm-matched random direction | 0 | 0.000 | 0.000 | 10.49 |
| Raw donor direction from counterfactual B | 25 | 0.781 | 0.875 | 0.098 |
| Smoke median-norm diagnostic | 18 | 0.562 | 0.656 | 1.37 |

P4 does not appear in the smoke: its PCA basis is fitted only in the
calibration stage. The P1 identity gate held bitwise. P5 at 0/32 with KL
10.49 and the raw donor at KL 0.098 with P0-agreement 0.875 show this site is
sensitive to patches on smoke data.

**Caution.** These are eight smoke groups: engineering data demonstrating the
machinery runs end to end on the released weights, not scientific
measurements. P0 accuracy 0.781 on the smoke is below the 80% pilot-usability
floor in the frozen §8 proposal; that is an early signal for the pilot
decision, **not** a smoke failure and **not** a pilot outcome — the pilot
evaluates the frozen thresholds on 128 disjoint pilot groups with
whole-group bootstrap uncertainty, and the human
[pilot decision document](../protocols/pilot_decision.md) records the outcome.
No compression, semantic, or scientific claim follows from this table.

Measured smoke resources (first full-model measurements on this stack; smoke
stage only, not a pilot throughput estimate):

| Measurement | Value |
|---|---|
| Fetch + hash verification of all pinned artifacts | 42.1 s |
| Target load + 32 identity gates | 120.4 s (318 forward calls) |
| AV load + description generation | 524.0 s (4,525 forward calls) |
| AR load + reconstruction | 67.1 s (32 forward calls) |
| Target reload + behavior scoring | 244.6 s (1,861 forward calls) |
| GPU peak allocated / reserved | 15,648,232,960 / 18,083,741,696 bytes (15.65 / 18.08 GB) |
| Runner wall time through bundle | 1,000.6 s (16.7 min) |
| Verified artifact bytes | target 15,242,805,751; AV 15,247,195,731; AR 10,920,058,064 |

## Concrete remaining blockers and limits

* Released-model **calibration and pilot are NOT RUN**. `baseline_fit_identity`
  is unfilled, every pilot decision field is PENDING PILOT RUN, and locked
  validation is not implemented in this runner.
* PyTorch's CUDA 13 wheel still warns about GB10 capability 12.1 versus its
  listed maximum 12.0. The completed smoke shows this exact workload ran once
  on this stack; the warning stands, and the calibration/pilot runs must
  confirm their own stages rather than inherit this smoke as a general
  compatibility certificate.
* No scientific threshold has been assessed: the 80% P0 floor, the 5-point P2
  loss bound, and the 32-group edit floor are design choices to lock before
  validation, and P4-versus-language and frozen-rule edit coverage are
  unmeasured on real models.
* The pilot refuses to start without `--calibration-fit` pointing at a
  completed calibration run; the two commands below must run in order.
* The pilot reports a projected 512-group validation cost against the
  eight-hour budget but never auto-starts validation; the projection must be
  checked against the budget before any validation decision.
* The suffix probe script and raw numbers are local scratch under
  `/tmp/owl_suffix_probe/` (not committed, not durable); the drift bound they
  justified is frozen in code and manifest, and the smoke rerun independently
  records 336 drift samples.

## Files and repository state

Branch: `phase-two-milestone-two`, based on
`e8587b5d61effefe351199aece5447b43ab8039b` (the Milestone 1 handoff commit).
This report records the local checks performed before the handoff commit. Use
the delivery branch rather than `main` to obtain the implementation. No merge,
visibility change, or weight publication is part of this handoff.

Git status at handoff, grouped:

| Changed area | Files |
|---|---|
| Claim ledger / project README (modified) | `docs/claims.md`, `research/open_weight_lingua/README.md` |
| Suffix-causality repair and stage machinery (modified) | `src/open_weight_lingua/target.py`, `runner.py`, `artifacts.py`, `audit.py` |
| New Milestone 2 modules (untracked) | `src/open_weight_lingua/splits.py`, `controls.py`, `text_edits.py`, `stats.py` |
| Tests (one modified, five new) | `tests/test_target.py`; `tests/test_splits.py`, `test_controls.py`, `test_text_edits.py`, `test_stats.py`, `test_pilot_stages.py` |
| Configs/protocols/scripts (new) | `configs/pilot.yaml`, `protocols/pilot_decision.md` (PENDING PILOT RUN template), `scripts/run_calibration.sh`, `scripts/run_pilot.sh` |
| Handoff | this report |

All paths above live under `research/open_weight_lingua/` except
`docs/claims.md`. Generated environments (`.venv-phase2/`), model caches
(`model-cache/`), run directories (`runs/`, including both smoke run dirs)
and `*.safetensors` are ignored and are **not** staged or committed. Phase One
runtime source, protocols, sealed results, weights, and completed conclusions
are unchanged.

## Next single DGX commands

From this checkout's repository root, in this order. Both reuse the already
hash-verified 41.41 GB cache in the main checkout; every artifact hash is
re-verified on load, and each invocation gets a fresh run directory.

```bash
bash research/open_weight_lingua/scripts/run_calibration.sh \
  --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache
```

then

```bash
bash research/open_weight_lingua/scripts/run_pilot.sh \
  --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache \
  --calibration-fit research/open_weight_lingua/runs/<calibration-run-dir>
```

Expected fresh-run outputs:

* Calibration (256 groups, target only): `manifest.json`, `results.json`,
  `summary.json`, `report.md`, `baseline_fit.safetensors`, `baseline_fit.json`
  (with the sha256 fit identity that `completion.json` records), plus
  `compatibility.json`, `completion.json`, `inventory.json`, `reports.zip`,
  local `wall_clock.json` and local `raw/*.safetensors`.
* Pilot (128 groups, requires the calibration fit): the same set minus the
  baseline-fit files, with `baseline_fit_identity`, the frozen edit-rule
  version and sha256 in the manifest, machine-readable keep/stop decision
  fields in `completion.json`, a decision section in `report.md`, and the
  projected 512-group validation cost against the eight-hour budget.
  Validation is never auto-started.

Guarantees that remain untested: real-model P4 PCA fit quality and its
byte-budget accounting; frozen-rule edit eligibility/coverage on the 128 pilot
groups; the §8 pilot-usability, preservation and edit-floor thresholds; every
bootstrap decision statistic on real models; pilot-stage stage times and
memory; and the projected validation cost. The smoke says nothing about any
of these.

## Addendum 2026-09-21 — first pilot attempt failed at the identity gate; kernel-shape pinning repair (pre-pilot)

This addendum preserves a failed run and its repair explicitly, per the
project honesty rules. It changes no earlier section; the unpadded smoke and
calibration evidence above is **superseded** by the pinned re-baselines whose
run IDs are recorded below after the rerun.

### What happened

* The 256-group calibration completed on the unpadded code: run
  `calibration-20260921T221636Z-63a6be98`, status COMPLETE, fit identity
  `74eca5c2f1d91b34eb328e613036d2f81d296a410ae357023af610c21d2c7d45`, frozen
  calibration median norm 98.43028259277344.
* The first pilot attempt, run `pilot-20260921T222056Z-69535f26`, **FAILED**
  in `Target.identity_gate` on the first variant, pilot-0000-A-x, with
  `RuntimeError: raw restoration changed greedy answer`. The run died before
  any condition executed (1 failed group, 127 skipped in `summary.json`);
  its `completion.json` says FAILED and its decision fields are degenerate
  artifacts of a dead run, not measurements. The run directory is retained
  untouched. **No pilot or validation outcomes were inspected**; the
  diagnosis below measures gate internals only.

### Diagnosis measurements (real model, gate internals only)

Replication was exact: the probe rebuilt the pilot plan and tokenization and
matched the failed run's plan hash and its `inputs[0]` row field-for-field.
On pilot-0000-A-x (64-token prompt, site layer 20 position 63):

| Measurement | Value |
|---|---|
| Gate outcome (reproduced) | `RuntimeError: raw restoration changed greedy answer` |
| Unpatched greedy trajectory | `[16, 15, 151645]` → `10<|im_end|>` (terminates) |
| Pinned-original greedy trajectory | `[16, 15, 198, 151643, 33975, …]` → `10\n<|endoftext|>Human Resources…` |
| First divergence step | 2 (sequence length 66), same two candidates swapped |
| Top-2 margin at divergence, unpatched / pinned | 0.75 (EOS over `\n`) / 0.375 (`\n` over EOS) — near-tie scale |
| **Unpadded prefix-site drift at the divergence step** | **2.58e-1 relative — beyond the frozen 1e-1 bound** |

Characterization (same variant unless noted; same-length repeatability
bitwise at lengths 64 and 66; capture settings ruled out bitwise):

| Check | Result |
|---|---|
| BF16 append sweep +1…+5, pilot-0000-A-x | 0.260 / 0.258 / 0.265 / 0.256 / 0.259 |
| Same sweep, pilot-0000-A-y | 0.158 / 0.144 / 0.159 / 0.159 / 0.165 |
| Same sweep, pilot-0001-A-x | 0.019 / 0.123 / 0.127 / 0.123 / 0.124 |
| Per-block growth 64→66 (layers 0/5/10/15/20) | 0.063 / 0.065 / 0.104 / 0.084 / 0.258 |
| **fp32 reload, same sweeps and layers** | **~3.2e-6 everywhere — collapse** |

Interpretation: the flip is a near-tie argmax perturbed by cross-length BF16
kernel-reselection noise (sm_121); fp32 collapse shows the prefix is
mathematically length-independent, so this is numerical noise, not a
semantic leak. But the noise has a **prompt-conditioned heavy tail** that
the smoke-based justification sample (336 measurements, max 3.55e-2; smoke
and pilot share the same length regime) badly under-sampled: all three
spot-checked pilot variants exceed the frozen bound at answer-suffix
lengths, so `score_answer`'s bound check would have failed in the behavior
stage too. The frozen bound's coverage claim is therefore **falsified for
the pilot distribution** (recorded in docs/claims.md).

### Why re-fitting the bound was rejected

The pre-approved repair ("same frozen bound, no new tolerance") was premised
on the drift being within 1e-1; it is not. Re-deriving a larger bound from
pilot-distribution measurements would fit a tolerance to the very data that
broke it and weaken the bound's wrong-vector-class discrimination. No
pilot/validation outcomes were inspected (gate internals only), so a
re-justification would have been permissible under brief §4 — but the
chosen, stronger repair removes the drift class by construction instead.

### The repair: kernel-shape pinning (`TARGET_BUCKET = 128`)

Every target-model forward in every stage (smoke/calibration/pilot) is
right-padded to one fixed length of 128 (zeros, `attention_mask` 0 on pads).
Rationale, each link previously measured: masked pads receive exactly zero
weight (finite-minimum additive bias before a float32 softmax — the
bitwise-proven same-length control); same-shape forwards are bitwise
deterministic on this backend; fp32 shows the prefix is
length-independent (~3.2e-6). With all GEMM shapes pinned, the unpatched
prefix vector is length-stable by construction. `TARGET_BUCKET` is a frozen
module constant in `target.py`; greedy and `score_answer` write generated
tokens/answer suffixes into successive masked pad slots (argmax and log-prob
positions derive from the mask); site semantics are unchanged. The runner
checks loudly before any inference that every tokenized input plus the
generation ceiling and answer suffix fits 128 (max observed prompt 79;
79 + 8 + 3 = 90); B never changes silently. `SUFFIX_DRIFT_BOUND` stays
frozen as an untouched backstop (never re-fitted), the same-length
dummy-suffix bitwise gate is unchanged, and the greedy identity comparison
(identity stage and behavior-stage P0/P1) is now a recorded backstop:
`exact`, or `drift_diverged` with measured evidence, counted per stage in
`summary.json` (`greedy_identity`), never hidden and never a decision input.
The manifest records `target_bucket`, `target_padding_policy`, and
`greedy_identity_gate` verbatim. AV/AR models are separate and untouched.

### Verification of the repaired code

| Check | Result |
|---|---|
| CPU suite (worktree venv) | **128 passed** (115 prior + 13 new; one existing expectation adjusted for the padded contract, justified in-test) |
| `uvx --from ruff==0.14.14 ruff check` on src/tests/scripts | PASS |
| `git diff --check` | PASS |
| Real-model gate check, pinned code (pilot-0000-A-x, pilot-0000-A-y, pilot-0001-A-x; gate internals only, scratch `/tmp/owl_greedy_probe/gate_check.json`) | identity gates bitwise, greedy status "exact" on all three, `suffix_drift_relative` exactly 0.0 |

### Superseded evidence and the pinned re-baseline

The unpadded runs `smoke-20260921T211151Z-a4e4a038` and
`calibration-20260921T221636Z-63a6be98` remain valid measurements of the
unpinned regime and are retained, but as pilot inputs they are superseded:
the pinned regime changes every target forward's numerics at BF16-noise
scale, so smoke, calibration and the pilot must all rerun on the pinned
code. New run IDs (to be filled by the operator after execution, replacing
each `PENDING RERUN` marker):

* Pinned smoke: **PENDING RERUN** —
  `bash research/open_weight_lingua/scripts/run_smoke.sh --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache`
* Pinned calibration: **PENDING RERUN** —
  `bash research/open_weight_lingua/scripts/run_calibration.sh --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache`
* Pinned pilot (consumes the pinned calibration fit): **PENDING RERUN** —
  `bash research/open_weight_lingua/scripts/run_pilot.sh --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache --calibration-fit research/open_weight_lingua/runs/<pinned-calibration-run-dir>`

The bounded stopping rule is unchanged: one pilot, at most one locked
validation, then a write-up even on failure. The pilot decision document
fields remain PENDING PILOT RUN.
