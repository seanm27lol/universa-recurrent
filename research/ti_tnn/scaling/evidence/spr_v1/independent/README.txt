Independent verification of the semantic-proposal-repair study (outputs copied by the producer session; the verifier
and its run are the reviewer's). The reviewer's session ended on a usage limit after the run, before it wrote its
summary report. The outputs below are the complete run.

The verifier: independent/verify_spr.py, sha256 81935f4be0f3ed16... (the script on disk; the run postdates its
last write), run at 1289d7e from the scratchpad: 167 checks, 1 failure.
  0_discipline 1, 10_bundle 6, 1_inventories 19, 2_lineage 36, 3_selection 6, 4_scores 8, 5_gold_isolation 6,
  6_preservation 15, 7_paired 16 (1 failure), 8_costs 29, 9_freeze 25.

The one failure: analysis.json's per-comparison `threshold` and `all_conditions` fields come from the shared
perf_analysis.paired helper (a threshold of 7 of 128, the performance study's rule scaled to two seeds). The
study's rule is +8, positive in both seeds and leaving any one database out (spr_run.LEAD), and the recorded decision
uses it. With pooled differences of +2, +2 and +4, neither threshold changes any outcome. analysis.json is kept as
written; the report says which fields to read.

score_mapping.jsonl: every scored record (run 384, dev 96) mapped to its snapshot sha256, the final SQL and the
reviewer's own EX. preservation_own.jsonl: the reviewer's own preservation re-execution.
