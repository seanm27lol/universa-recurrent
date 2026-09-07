# Claim ledger

| Statement | Status | Evidence or next test |
|---|---|---|
| Demo runs on a CPU | Implemented; see local validation record | CLI smoke tests |
| Reuses reduced numerical state | Implemented classical recurrence | `recurrence/solver.py` |
| Avoids repeated SVD during updates | Implemented | Factorization-blocking test |
| Candidate ambiguity leads to refusal | Implemented | Same-kernel candidate test |
| Projection/final witnesses reject tested corruptions | Tested for listed cases | Tamper, shape, nonfinite, and chain tests |
| Full-trace checker does not rerun a solve | Tested | Tests disable numerical solvers/factorizations |
| Compact witnesses prove the whole history | FALSE by design | Endpoint-only scope test and path counterexample |
| Upstream Universa integration | Optional adapter | Pin and adapter test; local validation reports whether exercised |
| Learned structure router | Not implemented | Future matched-evidence experiment |
| Learned neural recurrence | Not implemented | Future shared-update training and ablations |
| General cross-structure transport/discovery | Not integrated | Future typed contract and refusal tests |
| Efficient witness compression at useful scales | Untested research hypothesis | Measure overhead and property preservation |
| Faster inference than the strongest baseline | Not established | Direct/iterative baselines, full end-to-end costs |
| Faithful interpretation of opaque neural decisions | Not established | Causal interventions on an actual neural model |
| General performance gain from topology or category theory | Not claimed | Narrow task-specific evidence only |

Local tests are engineering validation, not a preregistered scientific result.
The exploratory benchmark includes toy dimensions and must not be presented as a
hardware or ML performance claim. The earlier repositories' results do not count
as results of this integration.
