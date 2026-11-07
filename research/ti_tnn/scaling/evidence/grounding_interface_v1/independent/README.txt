Independent verification of the grounding-interface study (outputs copied by the producer session; the verifier and
its run are the reviewer's).

The verifier: independent/verify_gi.py, sha256 b18c388221a54e97..., run at 429694a from an empty output directory.
It ran CPU only (cores 0-4,10-14), nothing written under the runs root, about 31 minutes. 295 checks, 0 failures:
  bundle 9, pins 19, prior_use 6, coverage 60, worker_items 2, permissions 36, scores 12, fallback 12,
  consumption 16, integrity 12, timing 61, constraint 14, dev 19, analysis 16, producer_code 1.

The main process imports no producer module. Producer code ran only in separate processes: a worker on the frozen
9c139ac code (schema items, the prompt's schema text, cached states into encodings, checkpoint loading and decoding)
and `python -m scaling.gi_report`.

Findings beyond the checks (none changes a result):
  - 1,149 of 1,152 eval decodes re-decode equal on the CPU with the reviewer's own bias vectors. The 3 differences
    are G0 near ties (margins of 0.0008-0.0072), likely TF32 on the GPU; the GPU's own reconstruction was equal.
  - Two constraint queries time out at 30 s on the efficiency cores but finish (incorrect, as recorded) with a 120 s
    deadline: host speed, not the producer.
  - analysis.json's gpu_hours is keyed by command name and drops the first dev ground reservation (0.302 h); the
    report lists all six.
  - The first dev ground run started before e759e58; the gate applies to the decodes, which both came after.
  - The reviewer's sqlglot reading of the record diagnostics is within tolerance of the analysis's (table recall
    0.960 against 0.958), not exact.
Not checked: GPU timings and generations (only their consistency); prompts by byte (by exact token count on 256);
Glex re-implementation; the sql_to_ir-based record diagnostics exactly.
