# Lingua v0: say what ran, and what can be checked

**Plain-language example:** “Fit the noisy measurements, while restricting the
estimate to balanced flows.” This is an explicit operation with a mathematical
meaning, not a speculative narration of a neural model's thoughts.

## The record contract

`universa-recurrent.trace.v1` records the supplied problem, operation sequence,
solver budget, output, and final multiplier witness. Numbers must be finite.
Unknown schemas and malformed shapes fail closed. No automatic conversion of
untrusted JSON into code is performed.

| Mode | Retained evidence | Independent checker establishes |
|---|---|---|
| compact | Per-step summaries plus problem and final witness | Final feasibility and optimality within tolerance; NOT the intermediate history |
| full | Compact fields plus each proposal, output, and projection multiplier | The retained numerical update chain plus the final claim, within tolerance |

Compact mode still keeps O(K) small step summaries for K iterations. It is not a
constant-memory streaming recorder. Full mode retains vectors at each step and
therefore costs more as state size grows. Both modes also carry the supplied
problem matrices; those bytes must be counted in storage comparisons.

Full mode also checks update chaining and objective values. Intermediate
stationarity values are producer diagnostics rather than independent conditioning
certificates. Compact summaries remain unverified telemetry even though the final
witness passes. The tests intentionally demonstrate this limitation.

## Do not confuse five different claims

| Claim | What would support it? |
|---|---|
| This vector satisfies the specified constraint | A constraint check |
| This vector solves the specified optimization problem | An optimality witness |
| These retained intermediate states form the claimed update sequence | Chaining checks |
| This remote machine really executed that sequence | A separate trusted recording/attestation system |
| These symbols explain the mechanism of a learned model | Held-out predictive and causal intervention evidence |

Only the first three are addressed here, numerically. The checker does not prove
the recorded problem was the one a remote user intended: verification must receive
trusted problem data when used outside this local demonstration.

## Why compact is not always enough

The paths 0→1→0 and 0→0 share endpoints. Only the first visits a positive state.
A final-only certificate cannot answer that historical question. A shorter valid
route is not necessarily the route that physically ran.

## Extension rule

For each new operation: give a familiar example, input/output types, precise
semantics, assumptions, the witness, an independent checker, measured costs, and
what remains unexplained. Keep the human-readable rendering deterministic where
possible. A learned explanation generator must never quietly replace evidence.

## Relation to Applied CMCM

The original project studies when computational records matter. This new repo
will compare which portions of the record suffice for particular checks, with
full records as a reference. That is a proposed experiment, not a consequence of
its earlier results. [Sources and boundaries](real_world_connections.md).
