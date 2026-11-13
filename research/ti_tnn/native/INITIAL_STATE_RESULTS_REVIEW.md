# Independent review of the completed initial-state study

Showing eight initial states did not produce the required complete-answer gains.
Two read-only local reviewers independently checked the saved evidence and found
no blocking scoring, endpoint, identity or resource-accounting defects. This is
independent checking of one local study, not an externally replicated learning run.

Reviewed training source: `3808e8209b4b22e36eded29343721195ca682ca4`.
Protocol SHA256: `bd108e03f51aefbc5bf8ce1cc0086c874d450b79ecf5668a23b275d1f37f50df`.
The [results](INITIAL_STATE_RESULTS.md) and [full tables](INITIAL_STATE_DETAILS.md)
retain the failed screen and all nine populations separately.

## Saved-answer review

An independent reference executor and string parser checked all 9,216 saved
completions and 36,864 register positions. Complete-answer, per-register,
changed/unchanged and per-register changed/unchanged scores match the saved
scores exactly. All 1,960 malformed outputs fail all four registers; no malformed
example disappears from a main denominator. All 321 prepared artifacts and 232
completion artifacts match their bindings.

The unchanged development screen recomputes to false for both recommendations.
Dense improves only +2/+2 on the two required value populations; shared improves
0/+1. Both would need +8/+8. Multi-state training anchors score 0–2/128, including
0/64 on individual transitions for every architecture. All 1,024 longer-program
answers are malformed. All eight high-initial transition populations score 0/399
on unchanged registers.

Shared in-range transition gains occur among unchanged positions, while changed
positions fall 24→23/114. Dense improves both. Heldout numeric CE decreases in all
architectures and populations but does not override the failed complete-answer
screen or establish a neural execution mechanism.

Private independent scoring report SHA256:
`1b04071a2877eedd195117cb8ec22e199ff9b3f0aa6c19ce1288b5f88febe733`.

## Endpoint, identity and budget review

The second reviewer verified all 51 source-file hashes, 321 prepared-artifact
hashes, 32 journaled checkpoint files/manifests and eight final checkpoint payloads.
Model, optimizer, scheduler, sampler, configuration, source, training/development
data and order identities match. Initial model hashes match across conditions
within each architecture. Every arm has exactly 1,424 contiguous updates, cursor
11,376 and 2,590,592 full-label tokens; no replayed updates or recovery occurred.

All 72 populations have complete CE coverage: 2,176 answer/EOS tokens, consisting
of 512 numeric tokens and 1,664 formatting/EOS tokens. State-readout row/pass/slot
coverage matches the prepared arrays, including 128 document starts per population.
The saved bounded CPU FP32 reports pass scale and strict-allclose checks in all
eight arms, with maximum relative RMS error `3.1074308909819825e-07`. Applicable
routes and bounded generated tokens match exactly. Floating outputs are not
bitwise identical. These reports are not general BF16-serving equivalence and
do not supersede historical strict-allclose failures.

The prelaunch ledger prefix is byte-preserved. Exactly one comparison reservation
closed normally, charging 3,331.3901202119887 seconds, or 0.9253861445033302 hours.
Native use is 7.146698220701847/12 hours, leaving 4.853301779298153 hours. The four-hour
allowance was part of that existing cap. There were no failed study attempts or
incomplete arms. A reviewer-script formatting error involving a null route-RMS
field was corrected locally; its failed audit log remains preserved and is not
misreported as a study failure.

Private endpoint/budget review SHA256:
`b60339570139834effd8fc0f0e538af3786ac49c20358dd7422800823ab282a4`.
Machine-readable audit SHA256:
`1ccb61f41233a7eb05c72baa70a38f6a9646f7770e7d354b7946ab8640013e44`.
Supplement SHA256:
`7dc117992ca80bd0bbd259a365b8f125e83ab34efc85aaebac4bce23e5e7643e`.

Neither reviewer changed producer files, trained a model, generated outputs or
checkpoints, or opened a reservation. The endpoint reviewer loaded saved checkpoint
payloads for validation only. CE, state and numerical reports were checked for
identity and coverage, not regenerated. The implementation owner's separate
saved-output reconstruction also matches all 72 complete score objects exactly.
Hashes identify local artifacts; they do not authenticate a remote execution.
