# Handoff: what the open-weight NLA assay has shown, and what to do next — 2026-10-04 (updated after Phase Four)

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The answer is 5. Just before the model writes it, we capture one internal
activation and do three things:

1. The released *activation verbalizer* (AV) describes the activation in
   English.
2. Its paired *reconstructor* (AR) turns that English back into a vector.
3. We put the vector back and see whether the model still answers 5. We also
   edit the English and see whether the answer follows the edit.

That is the whole assay: one site, one toy task, released Natural Language
Autoencoder (NLA) pairs for three model families. This page says what it has
established, points to the evidence, and ranks the next steps. Every claim
here is in [docs/claims.md](../../docs/claims.md) with its run IDs and
boundaries.

## What we can say

1. **Descriptions are informative but lossy.** Patching in the description's
   reconstruction (P2) keeps the model's behavior far closer to the original
   than a shuffled description (P3) does, on every family. But on every
   family with a usable task it loses more accuracy than the frozen 5-point
   bound. A generic PCA vector at the same byte budget (P4) does as well or
   better.

   | Family, site | Unmodified accuracy (P0) | P2 | P2 loss, 95% upper | P2 − P3 log-prob, lower | P4 |
   |---|---:|---:|---:|---:|---:|
   | Qwen2.5-7B, block 20/28 | 0.811 | 0.564 | 41.4 pp | +3.80 | 0.803 |
   | Gemma-3-12B, block 32/48 | 0.951 | 0.934 | 10.16 pp | +2.61 | 0.953 |
   | Gemma-3-27B, block 41/62 | 0.965 | 0.953 | 6.25 pp | +1.84 | 0.965 |

   The Gemma rows use the corrected answer instrument (see 4). All three
   pilots stopped under their frozen rules.

2. **The descriptions are next-token predictions, not statements about
   program state.** At the capture site every family writes the same genre,
   ending in "Final token … expecting "10" or "11"". On both Gemma pairs that
   guess is usually right (it leads the list in 108/128 and 111/128
   receivers), even for values the prompt never shows. But no description
   ever says "x is currently 10", and the other variable's value is absent.
   Capturing at the end of the program instead names the variables but not
   their values. So the frozen edit hypothesis — *edit a stated variable value
   and watch behavior follow* — is untestable with these released AVs. Edit
   coverage is 0/128 on all three families.

3. **On Gemma-3-12B the number stated in an AV description sets the answer,
   but only inside AV-written text — and the text need not describe this
   activation.**

   | What is patched in (Gemma-3-12B) | Answer moves to the written number |
   |---|---:|
   | The AV's description, every mention of the value rewritten | 42/121 fresh receivers (25/64 on the pilot split) |
   | The same, only the "Final token" candidates rewritten | 5/121 (2/64) — the rest of the text contradicts the edit |
   | Hand-written texts asserting the number (four frozen texts) | 0–1/121 each |
   | **Another program's AV description, every mention rewritten** (D5) | **46/121** |
   | That other description, unedited | 3/121; it pulls the answer to *its own* number in 39/106 |
   | Half the mentions, or every mention except the slot (D6) | 0/121; ¾ of them 11/121 |

   Any written value works: the counterfactual, the other neighbour, or a
   distant number. So the AR maps AV-register descriptions of this task to
   on-task directions in which the stated number sets the answer.
   Hand-written mimicry falls off that manifold and breaks the answer's
   format. Content specific to the individual activation is not needed. The
   preference for the written number grows smoothly with how much of the text
   states it, but the answer flips only when nearly the whole description
   agrees (D6). On Gemma-3-27B every edit shifts the log-probability
   toward the written number but almost never flips the answer (0–1/74). On
   Qwen, the closed steering assay's additive recipe was non-specific
   disruption.

4. **The models don't seem to keep a readable state at all (Phase Three,
   Stage 0).** We tried to build the missing state record ourselves.
   - A frozen survey of five positions and every layer found no place where
     Gemma-3-12B or Qwen2.5-7B linearly holds both variables' values. The
     best site, the answer position, reaches 0.50 / 0.46 on Gemma against a
     0.80 gate.
   - Split by the variable asked, that site reads the asked value at
     0.93 / 0.88 and the other at chance.
   - The values appear only after the question, late in depth. The models
     seem to compute the answer on demand, which is why Phase Two's
     describers predicted answers.
   - Phase Three closed at its first gate; its validation splits are still
     unopened. See [reports/phase_three_stage0.md](reports/phase_three_stage0.md).
   - Phase Four then checked every statement boundary, while the program is
     being read. Literals on the current line read well; arithmetic results
     read at about 0.2; carried computed values about 0. There is no running
     state either. See [reports/phase_four_trace.md](reports/phase_four_trace.md).

5. **Measure the instrument first.** The Gemma pilots first "failed" the
   usability gate (P0 0.52 and 0.25) because the models answer `"5\n"` and
   the frozen metric rejected the trailing newline. A frozen amendment fixed
   the comparison, and both confirmation pilots reproduced the closed pilots'
   generations bitwise. On this machine every rerun has reproduced bitwise.

## What we cannot say

- **No evidence that descriptions read their activation.**
  - D4 showed hand-written text fails; D5 showed another program's
    description works as well as the receiver's own.
  - So the edit channel is a property of the AR, the target and the AV's
    output genre, not of how faithfully a description reads its activation.
  - An earlier draft of this handoff and the D4 write-up said
    "receiver-specific content". D5 refuted that; both are corrected.
- **No size ranking.** 12B flips and 27B does not, but the sites, NLA pairs
  and numerics differ: the 27B AV is float32-native and was served BF16,
  which is unauditable locally.
- **No generality.** One toy task, integers 0–19, one site per family, and
  no locked validation run on any family.
- **No relabeling.** Every closed outcome stands.

## Where the evidence is

| Question | Report | Ledger section in `docs/claims.md` |
|---|---|---|
| Qwen pilot and phase closeout | `docs/phase_two_results.md`, `reports/milestone_two.md` | Phase Two pilot outcome and phase closeout |
| Gemma port, 12B and 27B pilots | `reports/gemma3_pilot.md`, `reports/gemma3_27b_pilot.md` | Phase Two second-family port; Gemma-3-27B extension |
| The answer-instrument fix and confirmations | `reports/post_hoc_answer_lens.md`, `reports/gemma3_12b_confirmation.md`, `reports/gemma3_27b_confirmation.md` | Post-hoc answer lens and frozen answer-convention amendment |
| Why edit coverage is 0/128 | `reports/edit_eligibility_structure.md` | Post-hoc structure of edit eligibility |
| Edit-channel runs D1–D6 | `reports/edit_channel_diagnostics.md` | Edit-channel diagnostics |
| Phase Three site survey (closed at Stage 0) | `reports/phase_three_stage0.md`, `protocols/phase_three_brief.md` | Phase Three |
| Phase Four boundary survey (closed) | `reports/phase_four_trace.md`, `protocols/phase_four_brief.md` | Phase Four |
| Qwen steering assay (closed) | `reports/steering_assay.md` | Post-Phase-Two steering assay |
| vLLM backend (gate failed; keep eager) | — | Phase Two vLLM backend |

Paths without `docs/` are under `research/open_weight_lingua/`. Run
directories stay local (`runs/`, git-ignored); reports cite run IDs and
manifest hashes.

## What to do next, ranked

1. **Optionally add Phase Three to the paper** (`paper/nla_three_families.md`,
   maintained in a separate session).
   - The paper covers everything through D6 (PRs #23 and #24, merged
     2026-10-03/04).
   - Phase Three's Stage 0 is a natural explanatory section: the models hold
     only the asked value, late, which is why the released describers
     predict answers.
   - `tests/test_paper_numbers.py` fails if the paper cites a number that is
     not in a declared source. Add `reports/phase_three_stage0.md` to its
     `sources:` header before quoting it.
2. **Consider the edit-channel line closed.**
   - D5 (2026-10-03) answered the open question: the effect does not need the
     receiver's own activation.
   - D6 (2026-10-03) gave the mechanism: near-total agreement is needed.

   What remains is optional characterization, not a test of the hypothesis:
   - which features of AV text the AR needs — paragraph ablations of a
     foreign description; why hand-written mimicry fails;
   - whether a different mention order changes D6's threshold;
   - why 27B does not flip, which needs a new site or position and so a new
     protocol, kept out of Phase Two's records.

   Run these only if the paper needs them, each frozen first.

3. **If the state question is pursued again, change the task.** Phases
   Three and Four showed that, on this task, these models don't hold a
   readable state, neither while reading nor at the end. A new brief would need a task that forces state to be kept (for
   example several questions about one program), or per-line and nonlinear
   readouts. It must not reuse Phase Three's survey to pick sites after the
   fact.

**Not recommended:**

- **Locked validation:** preservation already fails, and the projected cost
  is 13.7 h (Qwen) to 53 h (Gemma-3-27B) against an 8 h budget.
- **More capture positions in search of state statements:** D2 was negative,
  and it edges toward the site-shopping the Phase Two rules forbid.
- **The 70B pair:** 141 GB does not fit in 121 GB of unified memory.

**Open housekeeping:**

- the official Gemma small-file bytes are unverified, pending gated access;
- the vLLM worker test is skipped locally (worker venv not built);
- ruff 0.16.8 flags about 80 pre-existing style findings. Past "ruff clean"
  claims use ruff 0.14.14.

## How to run things here

- **Code and branches.** `research/open_weight_lingua/`, on the
  `gemma3-port` worktree (`~/projects/universa-recurrent-gemma`). `main`
  merges it; the last merge is `c7ad73b`.
- **Discipline.** Every measurement is frozen first: protocol, predictions
  and code are committed and pushed *before* any model forward. Each runs
  once, and the result is reported whatever it is. A source bug permits only
  a regression test plus a visibly versioned rerun.
- **Launchers** (run from the repository root; each runs the CPU suite
  first):
  - `scripts/run_pilot.sh` for smoke, calibration and pilot;
  - `scripts/run_edit_diagnostics.sh` for D1 and D2;
  - `scripts/run_edit_replication.sh` for D3;
  - `scripts/run_edit_oracle_control.sh` for D4;
  - `scripts/run_edit_foreign_control.sh` for D5;
  - `scripts/run_edit_consistency_dose.sh` for D6.
- **Audits.** Each module also takes `--audit <run-dir>`, as does
  `python -m open_weight_lingua.audit <run-dir>` for pilots.
- **Tests and lint.**
  - `.venv-phase2/bin/python -m pytest -q -c
    research/open_weight_lingua/pyproject.toml research/open_weight_lingua/tests`
    (247 pass, 1 skip);
  - `tests/test_paper_numbers.py`;
  - `uvx --offline ruff@0.14.14 check research/open_weight_lingua`.
- **The GPU** (GB10, 121 GB unified memory). The 27B pipeline peaks at about
  57 GB reserved.
  - The user-level ollama service pins about 44 GB while idle:
    `systemctl --user stop ollama-user`, and start it again afterwards.
  - Other sessions' jobs are not yours to stop; ask the person who owns
    them.
  - Do not edit `src/`, `scripts/` or `protocols/` while a queued run has not
    yet written its manifest. Runs hash those files at start.
- **Costs measured here.** Gemma-3-27B pilot about 13 h; Gemma-3-12B pilot
  about 7.6 h; diagnostics 20 min to 2.5 h each.
