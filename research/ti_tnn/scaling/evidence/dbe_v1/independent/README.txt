Independent post-run reconstruction of the database-evidence revision run (dbe).

Reviewer: the independent reviewer agent; the producer did not write or edit the verifier.
Verifier: research/ti_tnn/scaling/independent/verify_dbe.py
  sha256 d71704da35397982b572f1a036f7e91cb0ea8389128309d12cace5b475e502f4, run at branch head c14764d
  (the bundle's run records are those committed at 443a5b3, unchanged since).
Result: 165 checks, 0 failures, over:
- scores: all 384 SQL re-executed against gold on the pinned snapshots, with the reviewer's own EX rule;
- lineage, selection and fallback;
- lookup inputs: a fresh db-evidence-v1.4 lookup equals every recorded package;
- paired statistics, with the reviewer's own bootstrap and sign-flip;
- costs, ledger and co-tenancy;
- reference access;
- integrity;
- the report's numbers and statements.

Files (copied byte for byte from the reviewer's output directory; they contain machine paths, as recorded):
  verify_dbe.json          every check, with its evidence
  verify_dbe_summary.txt   the run log
  score_mapping.jsonl      each submitted SQL mapped to its snapshot sha256 and SQL sha256, with own scores
  LIMITATIONS.txt          what the reconstruction does not establish
  worker_job.json          input to the flagged separate-process worker (the producer's lookup and prompt inputs)
  worker_out.json          that worker's output

Reproduce:
  python research/ti_tnn/scaling/independent/verify_dbe.py --runs-root <runs root> --out <dir>
