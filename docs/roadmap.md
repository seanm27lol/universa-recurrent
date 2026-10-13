# Roadmap: earn each connection

## Phase-one status: complete

The structured-state experimental sequence closed on 2026-09-16 after its locked
fresh-input validation. The selected linear predictor met the declared
**limited-usefulness** criterion, not the separate **close-prediction** criterion.
See the [important findings](phase_one_results.md) and
[technical report](phase_one_technical_report.md).

| Completed part | What survived evaluation |
|---|---|
| Numerical recording and checking | Workload-specific setup/batching savings; remaining tail latency is reported |
| State descriptions | Approximate continuation preservation with explicit overflow fallback and counted costs |
| Typed mathematical edits | Specified field interventions remain closely aligned with direct edits |
| Response prediction | A useful but inaccurate local linear approximation; quadratic extrapolation can fail |
| Final validation | Locked primary, two fresh input blocks, explicit positive and negative criteria |

**No more GPU runs are required for this phase.** Existing scripts remain for
reproducibility, not as an automatic task queue. Documentation and a recorded
results demonstration close the sequence. The recorded demonstration is a replay,
not a live language model.

The broader goals below remain separate, uncompleted work. This closeout does not
establish discovery of new spaces, transport between spaces, opaque-neuron
semantics or unrestricted natural-language reconstruction. A future phase would
need its own task, baselines, budget and stopping rule.

## Phase-two status: complete, decision STOP

The open-weight transfer experiment closed on 2026-09-21 after its bounded
128-group pilot. A language description of one Qwen2.5-7B-Instruct
residual-stream vector, reconstructed through the released NLA pair, did not
preserve behavior within the frozen 5-percentage-point loss bound (P2 loss
upper estimate 41.4 pp); the frozen text-edit rule found 0/128 eligible
groups, so the edit hypothesis is untested; and the projected validation cost
(≈13.7 h) exceeded the eight-hour budget. Per the stopping rule the locked
validation was not run. See the [important findings](phase_two_results.md) and
[technical report](phase_two_technical_report.md).

| Completed part | What survived evaluation |
|---|---|
| Engineering smoke, calibration, pilot | All pinned runs COMPLETE with independent auditor PASS |
| Site sensitivity | Donor/random-direction controls show the site can affect the intended measurement |
| Language-route preservation | Not supported at the frozen bound; the description direction is on-task (beats the wrong-description control) but not preserving |
| Targeted text edits | Untested: zero explicit current-value coverage under the frozen rule |
| Numerics on GB10 | Same-length bitwise causality gate and kernel-shape pinning; the falsified pre-pilot drift bound is preserved on record |

**No more GPU runs are required for this phase.** A negative pilot does not
trigger checkpoint, site, layer or task shopping; a future phase would need
its own brief, budget and stopping rule.

**Since the closeout.** Each piece of follow-up work had its own frozen
protocol and recorded outcome, and none reopens this phase:

- Gemma-3-12B and 27B replications;
- an answer-instrument fix with bitwise-reproduced confirmation pilots;
- a structural account of the 0/128 edit coverage;
- seven edit-channel diagnostics. On Gemma-3-12B, consistently editing an
  AV description moves the answer to the stated number, even when the
  description is of a different program, but only when nearly the whole text
  agrees. Hand-written text does not.

The current state and ranked next steps are in the
[open-weight NLA handoff](../research/open_weight_lingua/HANDOFF.md).

## Phase-three status: closed at Stage 0, 2026-10-04

Phase Three tried to build the missing piece itself: a typed state record
read out of the model and edited causally. Its frozen site survey found no
position where Gemma-3-12B or Qwen2.5-7B linearly holds both variables'
values. Only the asked variable's value appears, late, after the question.
The gate failed, so the later stages never ran and the validation splits
stay unopened. See
[the Stage 0 report](../research/open_weight_lingua/reports/phase_three_stage0.md).

## Phase-four status: closed after its survey, 2026-10-04

Phase Four checked whether the state is tracked while the program is read.
At every statement boundary, the models hold the literal written on that
line. Results of arithmetic barely register, and values carried from
earlier lines barely register. Its gate failed, as predicted. With Phase
Three, the probes recovered no readable running state on this task. That
is a limit on these readouts, not proof that no state is kept. See
[the Phase Four report](../research/open_weight_lingua/reports/phase_four_trace.md).

## Phase-five status: closed after its survey, 2026-10-04

Phase Five asked whether the state is there but nonlinear, as with
Othello-GPT's board. A small MLP probe, under the same rules, read about
what the linear probes read: the literal on the current line, and the
asked value after the question. Its gate failed, as predicted. Later
review showed that the original readout cannot see a magnitude code (see
the readout audit), so the limit is on the tested probes. See
[the Phase Five report](../research/open_weight_lingua/reports/phase_five_nonlinear.md).

## Phase-six status: closed at Stage 0, 2026-10-05

Phase Six moved the question before the program and reworded it. The
asked and the not-asked variable read alike at every line within small,
unresolved margins. Of the positions probed, the asked value reads best at
the answer. Accuracy dropped and the usability gate failed, so the phase
closed. See
[the Phase Six report](../research/open_weight_lingua/reports/phase_six_question_first.md).

## Phase-seven status: exploratory factorial complete, 2026-10-05

Phase Seven crossed question position with wording on the same programs.
Qwen's Phase Six loss is a position effect. Gemma's is a smaller position
effect plus a wording × position interaction. See
[the Phase Seven report](../research/open_weight_lingua/reports/phase_seven_prompt_factorial.md).

## Phase-seven confirmatory: all six hypotheses supported, 2026-10-06

A 2×2 frozen before the reserved `validation_a` split was opened replicated
the attribution on unseen data. Gemma's wording effects came out smaller
than in the exploratory run. `validation_b` remains reserved. See
[the confirmatory report](../research/open_weight_lingua/reports/phase_seven_confirmatory.md).

## Phase-eight status: answer routing confirmed, 2026-10-06

Attention knockout located where each model's answer consults the program:
Qwen's answer position in layers 21–23, Gemma's question tokens in layers
24–29. All six claims frozen from the exploratory run held on
`validation_b`. Both reserved validation blocks are now spent. Which program
tokens are read (raw statements or stored state) is the open next question.
See [the Phase Eight confirmatory report](../research/open_weight_lingua/reports/phase_eight_confirmatory.md).

## Phase-nine status: the read targets the asked statements, confirmed, 2026-10-12

Partial-span knockout showed that the late read retrieves the asked
variable's own statements, and only those. No end-of-program summary is
needed. All six claims held on 256 freshly generated groups (seed 206001),
frozen before generation. Open next: when computed values reach the asked
statements. A claim about copied values was withdrawn on 2026-10-12: it
rested on a mislabelled breakdown.
See [the Phase Nine confirmatory report](../research/open_weight_lingua/reports/phase_nine_confirmatory.md).

## Phases ten and eleven: a degraded final assignment falls back, confirmed, 2026-10-12

Blocking the asked variable's final statement from earlier lines degrades
it, whichever lines are blocked, and the model then answers the previous
assignment. Confirmed on 512 freshly generated groups (seed 207001). Open
next: whether the late read selects the most recent *intact* assignment.
Degrade the last two assignments and look for a fallback to the third.
See [the Phase Eleven confirmatory report](../research/open_weight_lingua/reports/phase_eleven_confirmatory.md).

## Phase-twelve status: most recent intact assignment, confirmed, 2026-10-12

The late read takes the asked variable's most recent intact assignment.
Degrading assignments from the end pushes the answer back one assignment at
a time. Confirmed on 1,536 freshly generated groups (seed 208001). Across
Phases Eight to Twelve the account is a recency-ordered lookup at answer
time. Open next: whether it holds at larger scale (Gemma-3-27B).
See [the Phase Twelve confirmatory report](../research/open_weight_lingua/reports/phase_twelve_confirmatory.md).

## Phase-thirteen status: the account holds at Gemma-3-27B, confirmed, 2026-10-12

The larger sibling model routes the late read the same way: its answer
positions never need the program, its post-program positions read it late,
and it takes the most recent intact assignment. The read ends deeper: by
layer 46 of 62 (74% of depth) against 30 of 48 (62%) in the 12B. Confirmed
on 1,536 freshly generated groups (seed 209001). Open next: whether the read
changes when the question comes before the program.
See [the Phase Thirteen confirmatory report](../research/open_weight_lingua/reports/phase_thirteen_confirmatory.md).

## Phases fourteen and fifteen: the question-first read, confirmed, 2026-10-12

With the question before the program, the late read still uses the asked
variable's statements and still has no usable end-of-program summary, but
it chooses worse. It takes the other variable's statements, and for Qwen
also superseded ones. Hiding those distractors restores question-after
accuracy. Gemma's read also ends later. Seven of eight claims held on 512
freshly generated groups (seed 210001); the eighth narrowed "no summary" to
"no summary large enough to answer from". Open next: how the asked variable
reaches the late read when the question comes first.
See [the confirmatory report](../research/open_weight_lingua/reports/phase_fifteen_confirmatory.md).

## Phase sixteen: the program carries the asked variable, confirmed for Gemma, 2026-10-13

With the question first, Gemma marks the asked variable on the program as
it reads it. Cut the program off from the question and the late read takes
the most recent assignment of either variable. Qwen's variable is confused
by the same cut, but its fallback is not specifically recency. Five of six
claims held on 768 freshly generated groups (seed 211001). Open next: which
program tokens carry the mark: the asked statements, the other's, or the
program's end.
See [the Phase Sixteen confirmatory report](../research/open_weight_lingua/reports/phase_sixteen_confirmatory.md).

## Phase seventeen: the program marks both variables, confirmed, 2026-10-13

With the question first, the program tokens mark the asked variable's
statements and the other variable's statements; the late read needs both
marks. In Gemma the mark on the other variable is spread over its lines.
Five of six claims held on 768 freshly generated groups (seed 212001). Open
next: at which layers the program reads the question.
See [the Phase Seventeen confirmatory report](../research/open_weight_lingua/reports/phase_seventeen_confirmatory.md).

## Phase eighteen: marking precedes the late read, confirmed, 2026-10-13

With the question first, the program reads the question in the middle
layers and is finished by layer 24 of 48 (Gemma) or 17 of 28 (Qwen), before
the late read. Four of six claims held on 768 freshly generated groups (seed
213001); the marking after layers 18 and 14 was smaller than 5 points.
See [the Phase Eighteen confirmatory report](../research/open_weight_lingua/reports/phase_eighteen_confirmatory.md).

## Original project stages (not a new run list)

| Stage | Deliverable | Gate before a claim |
|---|---|---|
| 0 | Transparent classical solver and witnesses | Independent numerical checks |
| 1 | Neural v1: one early route, shared recurrence | Exploratory only; failure modes retained |
| 2 | **Neural v2: parallel hypotheses, route revision, commit/abstain** | Full controls, calibration/test separation, DGX benchmark |
| 3 | Typed state transport between structures | Check transport laws and quantify information loss |
| 4 | Structure discovery and admission | Novelty/certification tests and refusal controls |
| 5 | Compact Lingua budgets | Preserve declared checks while reducing storage/check cost |
| 6 | Neural semantic interpretation | Held-out predictive and causal interventions |

## Broader v2 questions retained for future scoping

1. Does delaying commitment reduce wrong-route damage on an untouched test seed?
2. Does the posterior mixture beat hard selection without privileged generator knowledge?
3. Does active compaction lower wall-clock cost at any realistic batch size?
4. Does v2 outperform direct, untied, ambient, and dedicated-depth controls?
5. Are its route probabilities sufficiently calibrated for the declared rejection rule?

A direct solver, a refusal, or a negative latency result can be the correct outcome.
