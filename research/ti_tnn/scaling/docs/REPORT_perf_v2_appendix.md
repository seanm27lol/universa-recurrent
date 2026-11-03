# Performance study v2: every question, and the full post-evaluation diagnosis (appendix)

Companion to [`REPORT_perf_v2.md`](REPORT_perf_v2.md). Both parts are generated from the committed evidence bundle
`evidence/perf_v2/` alone:

    python -m scaling.perf_report --questions
    python -m scaling.diagnose perf-tables --out scaling/evidence/perf_v2/diagnosis/20261103T092801Z_post_f9e98d4

The executed SQL of every request lives in the bundled records:
- the controllers: `eval/<seed run>/<arm>/records.jsonl`, field `final.sql`, with its score and execution status;
- the baselines: `baselines/<run>/records.jsonl`, whose `attempts` hold every turn, tool call and observation.

## Every question

Each controller cell is the arm's number of EX-correct seeds out of 3. One letter is added per affected seed:
- t: the submitted query timed out (30 s);
- e: it failed in SQLite.

Each baseline cell is 1 or 0 (a = abstained), with its tool calls in parentheses.

### historical population

| question | database | T4-v1 | C4-v1 | T4-v2 | C4-v2 | L | B4 | Q27 |
|---|---|---|---|---|---|---|---|---|
| bird23tf-04392 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | 0 (3) | 0 (3) |
| bird23tf-00788 | soccer_2016 | 0 | 0 | 0 | 0 | 0 (3) | a (8) | 1 (1) |
| bird23tf-05697 | hockey | 0 | 0 | 0 | 0 | 0 (3) | 1 (1) | 1 (1) |
| bird23tf-04901 | image_and_language | 0 | 0 | 0 | 0 | 1 (2) | 1 (1) | 1 (1) |
| bird23tf-01627 | simpson_episodes | 0 | 1 | 1 | 2 | 0 (6) | 1 (3) | 1 (1) |
| bird23tf-03130 | world | 0 | 0 | 0 | 0 | 0 (2) | 1 (6) | 1 (3) |
| bird23tf-02636 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (5) | 1 (0) | 1 (1) |
| bird23tf-06046 | professional_basketball | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 0 (5) |
| bird23tf-04982 | music_platform_2 | 0 | 0 | 0 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-03420 | software_company | 0 | 0 | 0 | 0 | 0 (3) | a (8) | 1 (3) |
| bird23tf-01859 | disney | 0 | 0 | 0 | 0 | 0 (2) | a (0) | 0 (2) |
| bird23tf-00469 | talkingdata | 0 t | 1 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-03943 | computer_student | 1 | 0 | 1 | 1 | 0 (4) | 1 (2) | 1 (2) |
| bird23tf-03526 | food_inspection | 2 | 2 | 0 | 2 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-04345 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-04097 | soccer_2016 | 0 | 0 | 3 | 1 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-03029 | hockey | 0 | 0 | 0 | 0 | 0 (3) | 0 (4) | 1 (1) |
| bird23tf-04905 | image_and_language | 1 | 1 | 2 | 3 | 0 (3) | 1 (1) | 1 (1) |
| bird23tf-01628 | simpson_episodes | 0 | 0 | 0 | 0 | 0 (3) | 1 (3) | 1 (1) |
| bird23tf-03134 | world | 2 | 2 | 3 | 2 | 0 (8) | 1 (7) | 1 (1) |
| bird23tf-02664 | ice_hockey_draft | 0 | 0 | 1 | 0 | 0 (7) | 1 (1) | 1 (1) |
| bird23tf-04254 | professional_basketball | 0 | 0 | 0 | 2 | 0 (3) | 1 (3) | 1 (1) |
| bird23tf-06518 | music_platform_2 | 3 | 2 | 3 | 2 | 0 (5) | 1 (4) | 1 (1) |
| bird23tf-03434 | software_company | 0 | 0 | 0 | 0 | 0 (1) | 1 (3) | 1 (3) |
| bird23tf-06228 | disney | 1 | 1 | 2 | 2 | 0 (1) | 0 (1) | 1 (1) |
| bird23tf-00443 | talkingdata | 1 | 0 tt | 3 | 2 | 0 (5) | 0 (5) | 1 (1) |
| bird23tf-05912 | computer_student | 2 | 2 | 1 | 0 | 0 (3) | 1 (1) | 1 (1) |
| bird23tf-03525 | food_inspection | 0 | 0 | 1 | 2 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-01565 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-00840 | soccer_2016 | 0 | 0 | 0 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-03022 | hockey | 0 | 0 | 0 | 0 | 0 (7) | 1 (3) | 1 (1) |
| bird23tf-02969 | image_and_language | 0 | 0 | 0 | 0 | 0 (3) | 0 (1) | 0 (1) |
| bird23tf-01625 | simpson_episodes | 1 | 0 | 1 | 0 | 1 (1) | 1 (1) | 1 (1) |
| bird23tf-04970 | world | 1 | 1 | 0 | 2 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-02659 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (3) | 1 (1) | 1 (1) |
| bird23tf-01167 | professional_basketball | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 0 (5) |
| bird23tf-06520 | music_platform_2 | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 0 (1) |
| bird23tf-03442 | software_company | 0 | 0 | 0 | 0 | 0 (1) | 0 (1) | 0 (1) |
| bird23tf-04521 | disney | 0 | 0 | 0 | 0 | 0 (3) | a (8) | 1 (1) |
| bird23tf-05284 | talkingdata | 0 | 0 t | 1 | 0 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-00417 | computer_student | 0 | 0 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-03528 | food_inspection | 1 | 0 | 1 | 3 | 0 (1) | 0 (1) | 0 (1) |
| bird23tf-01479 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 1 (1) |
| bird23tf-00809 | soccer_2016 | 0 | 0 | 1 | 1 | 0 (3) | 1 (1) | 0 (2) |
| bird23tf-05703 | hockey | 0 | 0 | 0 | 0 | 0 (1) | 0 (5) | 0 (1) |
| bird23tf-02994 | image_and_language | 0 | 0 | 0 | 0 | 0 (3) | 0 (2) | 0 (1) |
| bird23tf-01686 | simpson_episodes | 0 | 0 | 0 | 0 | 0 (5) | 0 (1) | 0 (1) |
| bird23tf-03127 | world | 2 | 1 | 2 | 3 | 1 (1) | 1 (2) | 1 (1) |
| bird23tf-02670 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (5) | 1 (3) | 1 (2) |
| bird23tf-01149 | professional_basketball | 0 | 0 | 2 | 1 | 0 (1) | a (8) | 0 (1) |
| bird23tf-06507 | music_platform_2 | 0 | 0 | 0 | 0 | 0 (6) | 1 (2) | 1 (1) |
| bird23tf-05748 | software_company | 0 | 0 | 0 | 0 | 0 (3) | a (8) | 1 (1) |
| bird23tf-04507 | disney | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 1 (2) |
| bird23tf-05928 | talkingdata | 0 | 0 | 0 | 0 | 0 (1) | 1 (3) | 1 (1) |
| bird23tf-03953 | computer_student | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 0 (4) |
| bird23tf-03551 | food_inspection | 0 | 1 | 3 | 1 | 0 (1) | 1 (3) | 1 (1) |
| bird23tf-01560 | public_review_platform | 0 | 0 | 0 | 0 | 0 (2) | a (8) | 0 (1) |
| bird23tf-00851 | soccer_2016 | 0 | 0 | 0 | 0 | 0 (8) | 1 (2) | 1 (1) |
| bird23tf-03037 | hockey | 0 | 0 | 0 | 0 | 0 (3) | 0 (4) | 1 (1) |
| bird23tf-02956 | image_and_language | 0 | 0 | 0 | 0 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-01640 | simpson_episodes | 0 | 0 | 0 | 0 | 0 (3) | 1 (3) | 1 (2) |
| bird23tf-03139 | world | 0 | 0 | 2 | 1 | 1 (1) | 1 (3) | 1 (1) |
| bird23tf-02663 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (1) | 1 (5) | 1 (1) |
| bird23tf-01158 | professional_basketball | 0 | 0 | 0 | 0 | 0 (1) | 1 (5) | 1 (1) |

### new population

| question | database | T4-v1 | C4-v1 | T4-v2 | C4-v2 | L | B4 | Q27 |
|---|---|---|---|---|---|---|---|---|
| bird23tf-06525 | music_platform_2 | 3 | 2 | 2 | 3 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-03426 | software_company | 0 | 0 | 0 | 0 | 0 (2) | 1 (1) | 1 (1) |
| bird23tf-01838 | disney | 0 | 0 | 0 | 0 | 0 (1) | 0 (2) | 0 (1) |
| bird23tf-05932 | talkingdata | 0 | 0 | 0 e | 1 | 0 (1) | a (8) | 0 (3) |
| bird23tf-05268 | computer_student | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 0 (3) |
| bird23tf-03519 | food_inspection | 0 | 0 | 0 | 0 | 0 (2) | 1 (5) | 1 (1) |
| bird23tf-01450 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | 0 (3) | 0 (1) |
| bird23tf-00784 | soccer_2016 | 1 | 0 | 2 | 0 | 0 (8) | 1 (1) | 1 (1) |
| bird23tf-04922 | hockey | 0 | 0 | 1 | 0 | 0 (3) | 1 (3) | 1 (3) |
| bird23tf-02962 | image_and_language | 0 | 0 | 0 | 0 | 0 (5) | a (8) | 0 (5) |
| bird23tf-04433 | simpson_episodes | 1 | 1 | 1 | 2 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-04963 | world | 0 | 0 | 0 | 2 | 0 (1) | 1 (6) | 1 (1) |
| bird23tf-02680 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (3) | 1 (2) | 1 (2) |
| bird23tf-01113 | professional_basketball | 0 | 0 | 1 | 1 | 0 (1) | a (8) | 1 (2) |
| bird23tf-03149 | music_platform_2 | 0 tt | 0 t | 0 | 0 | 0 (1) | 1 (3) | 1 (2) |
| bird23tf-03440 | software_company | 0 e | 0 t | 0 ee | 0 | 0 (6) | 1 (4) | 0 (3) |
| bird23tf-04524 | disney | 0 | 0 | 0 | 0 | 1 (1) | 1 (3) | 1 (1) |
| bird23tf-03980 | talkingdata | 0 | 0 | 1 | 0 | 0 (6) | a (8) | 1 (3) |
| bird23tf-03949 | computer_student | 0 | 0 | 1 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-03537 | food_inspection | 0 | 0 | 0 | 0 | 0 (5) | a (8) | 0 (1) |
| bird23tf-04385 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 1 (7) |
| bird23tf-04133 | soccer_2016 | 0 | 0 | 0 | 0 | 0 (8) | 1 (1) | 1 (1) |
| bird23tf-03099 | hockey | 0 | 0 | 0 | 0 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-02979 | image_and_language | 0 | 0 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-04436 | simpson_episodes | 2 | 3 | 2 | 1 | 1 (3) | 1 (3) | 1 (2) |
| bird23tf-03138 | world | 0 | 0 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-02661 | ice_hockey_draft | 0 | 0 | 0 | 1 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-01130 | professional_basketball | 0 | 0 | 1 | 1 | 0 (3) | 1 (2) | 1 (2) |
| bird23tf-06528 | music_platform_2 | 2 | 2 | 3 | 2 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-03445 | software_company | 0 | 0 | 0 | 0 | 0 (6) | 0 (2) | 1 (2) |
| bird23tf-04513 | disney | 0 | 0 | 1 | 0 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-00472 | talkingdata | 0 | 0 | 2 | 2 | 1 (1) | 1 (1) | 1 (1) |
| bird23tf-05910 | computer_student | 0 | 0 | 0 | 0 | 0 (6) | 1 (2) | 1 (2) |
| bird23tf-05138 | food_inspection | 1 | 1 t | 1 | 1 | 0 (1) | a (8) | 1 (1) |
| bird23tf-01576 | public_review_platform | 0 t | 0 | 0 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-00736 | soccer_2016 | 0 e | 0 | 0 | 0 | 0 (5) | 1 (1) | 1 (1) |
| bird23tf-03060 | hockey | 0 | 0 | 0 | 0 e | 0 (8) | 0 (5) | 0 (2) |
| bird23tf-02989 | image_and_language | 0 | 0 | 1 | 1 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-01670 | simpson_episodes | 0 | 0 | 0 | 0 | 0 (1) | 0 (2) | 0 (1) |
| bird23tf-03126 | world | 0 | 0 | 3 | 1 | 0 (6) | 1 (1) | 1 (1) |
| bird23tf-04824 | ice_hockey_draft | 0 | 0 | 0 | 0 | 0 (1) | 0 (1) | 0 (1) |
| bird23tf-06049 | professional_basketball | 0 | 0 | 0 | 0 | 0 (3) | 0 (2) | 1 (1) |
| bird23tf-06516 | music_platform_2 | 0 | 0 | 0 | 0 | 0 (3) | 1 (8) | 1 (1) |
| bird23tf-03432 | software_company | 0 | 0 | 0 | 0 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-01824 | disney | 0 | 0 | 0 | 1 | 0 (1) | 1 (4) | 1 (2) |
| bird23tf-05282 | talkingdata | 3 | 3 | 3 | 2 | 0 (1) | 1 (4) | 1 (1) |
| bird23tf-03938 | computer_student | 3 | 3 | 2 | 3 | 0 (1) | 0 (3) | 1 (1) |
| bird23tf-05134 | food_inspection | 3 | 2 | 3 | 2 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-01506 | public_review_platform | 0 | 0 | 0 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-00765 | soccer_2016 | 2 | 0 | 1 | 1 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-06499 | hockey | 0 | 0 | 0 | 0 | 0 (3) | 1 (4) | 1 (2) |
| bird23tf-02964 | image_and_language | 0 | 0 | 1 e | 2 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-01698 | simpson_episodes | 0 | 0 | 0 | 0 | 1 (3) | a (8) | 1 (3) |
| bird23tf-03121 | world | 0 | 0 | 1 | 1 | 0 (1) | 1 (1) | 1 (1) |
| bird23tf-02673 | ice_hockey_draft | 0 | 0 | 1 | 0 | 0 (1) | 1 (2) | 1 (1) |
| bird23tf-01110 | professional_basketball | 0 | 0 | 0 | 0 | 0 (1) | 0 (2) | 0 (1) |
| bird23tf-05715 | music_platform_2 | 0 | 0 | 0 | 0 | 0 (3) | 1 (2) | 1 (1) |
| bird23tf-05090 | software_company | 0 | 0 | 0 | 0 | 0 (3) | a (0) | 0 (2) |
| bird23tf-01847 | disney | 0 | 0 | 0 | 0 | 0 (1) | a (8) | 1 (3) |
| bird23tf-00451 | talkingdata | 0 | 0 | 0 | 0 | 0 (2) | 0 (1) | 0 (1) |
| bird23tf-00430 | computer_student | 0 | 0 | 0 | 0 | 0 (1) | 0 (1) | 1 (1) |
| bird23tf-03518 | food_inspection | 0 | 0 | 0 | 2 | 0 (2) | 1 (1) | 1 (1) |
| bird23tf-01546 | public_review_platform | 0 | 0 | 0 | 0 | 0 (2) | 0 (4) | 0 (2) |
| bird23tf-05321 | soccer_2016 | 0 | 0 | 1 | 2 | 0 (1) | 1 (1) | 1 (1) |


## The post-evaluation diagnosis, both populations (CPU; each fit under its own grounding)

### Diagnosis: historical population (192 requests per arm: 3 seeds)

| free decoding | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| correct | 18 | 15 | 34 | 33 |
| table set equal | 57 | 56 | 93 | 87 |
| every gold table predicted | 92 | 93 | 133 | 125 |
| every predicted table in gold | 107 | 115 | 118 | 108 |
| join keys equal given tables equal | 48 | 47 | 81 | 77 |
| correct given table set equal | 13 | 10 | 28 | 25 |
| correct given table set differs | 5 | 5 | 6 | 8 |
| exact plan | 4 | 4 | 2 | 7 |
| requests with same reference join | 31 | 30 | 0 | 0 |
| timeouts | 1 | 3 | 0 | 0 |
| EX-correct with a suspicious program | 8 | 6 | 8 | 12 |

| error category (requests; a request can have several) | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| projection | 121 | 128 | 116 | 111 |
| table grounding | 135 | 136 | 99 | 105 |
| predicate column | 116 | 119 | 111 | 106 |
| join key (different tables) | 111 | 112 | 91 | 95 |
| aggregation | 42 | 45 | 48 | 39 |
| predicate literal | 30 | 30 | 29 | 33 |
| join: undeclared, non-gold key | 33 | 38 | 20 | 25 |
| ordering or LIMIT | 26 | 27 | 25 | 22 |
| grouping or HAVING | 20 | 26 | 23 | 18 |
| DISTINCT | 16 | 20 | 23 | 24 |
| nested query involved | 15 | 18 | 17 | 19 |
| join: same-reference equality | 31 | 30 | 0 | 0 |
| join key (same tables) | 9 | 9 | 12 | 10 |
| predicate operator or shape | 2 | 1 | 0 | 4 |
| execution timeout | 1 | 3 | 0 | 0 |

| first divergence from the gold traversal | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| FROM table | 83 | 74 | 89 | 79 |
| join count or kind | 37 | 32 | 20 | 27 |
| gold not representable | 18 | 18 | 18 | 18 |
| joined table | 20 | 28 | 11 | 11 |
| WHERE predicate structure | 7 | 11 | 9 | 14 |
| ON column (operand 0) | 6 | 5 | 12 | 10 |
| WHERE literal | 3 | 7 | 7 | 12 |
| SELECT aggregate | 4 | 5 | 7 | 5 |
| exact | 4 | 3 | 2 | 6 |
| WHERE column | 4 | 1 | 4 | 3 |
| FROM structure | 1 | 2 | 1 | 3 |
| GROUP BY | 2 | 1 | 3 | 1 |
| SELECT structure | 1 | 0 | 4 | 1 |
| SELECT column | 1 | 3 | 0 | 0 |
| DISTINCT | 0 | 1 | 1 | 1 |
| HAVING column | 0 | 0 | 2 | 1 |
| ON column (operand 1) | 1 | 1 | 1 | 0 |
| ORDER BY | 0 | 0 | 1 | 0 |

| teacher-forced decision (error rate) | steps per arm | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|---|
| SELECT other | 519 | 0.12 | 0.11 | 0.10 | 0.10 |
| WHERE operator | 480 | 0.25 | 0.26 | 0.25 | 0.24 |
| WHERE column | 285 | 0.32 | 0.31 | 0.26 | 0.27 |
| WHERE literal | 279 | 0.47 | 0.48 | 0.48 | 0.45 |
| SELECT column | 261 | 0.42 | 0.43 | 0.33 | 0.34 |
| LIMIT | 201 | 0.02 | 0.02 | 0.01 | 0.01 |
| HAVING operator | 189 | 0.00 | 0.00 | 0.00 | 0.00 |
| DISTINCT | 186 | 0.08 | 0.09 | 0.12 | 0.11 |
| FROM table | 186 | 0.47 | 0.44 | 0.50 | 0.47 |
| GROUP BY other | 186 | 0.02 | 0.04 | 0.02 | 0.01 |
| QUERY | 186 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: no more tables | 186 | 0.25 | 0.23 | 0.23 | 0.25 |
| STOP | 174 | 0.00 | 0.00 | 0.00 | 0.00 |
| ON column (operand 0) | 126 | 0.36 | 0.31 | 0.36 | 0.37 |
| ON column (operand 1) | 126 | 0.23 | 0.17 | 0.02 | 0.02 |
| ON operator | 126 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: add table | 126 | 0.52 | 0.56 | 0.27 | 0.31 |
| SELECT aggregate | 78 | 0.60 | 0.60 | 0.58 | 0.59 |
| GROUP BY column | 27 | 0.63 | 0.78 | 0.74 | 0.78 |
| ORDER BY other | 24 | 0.62 | 0.42 | 0.54 | 0.54 |
| ORDER BY column | 15 | 0.20 | 0.47 | 0.27 | 0.33 |
| HAVING column | 3 | 0.67 | 0.33 | 0.67 | 0.67 |
| HAVING literal | 3 | 0.67 | 1.00 | 0.00 | 0.00 |
| mean step accuracy (representable requests) | 174 | 0.785 | 0.785 | 0.808 | 0.809 |

### Diagnosis: new population (192 requests per arm: 3 seeds)

| free decoding | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| correct | 21 | 17 | 36 | 35 |
| table set equal | 57 | 54 | 82 | 81 |
| every gold table predicted | 90 | 87 | 119 | 117 |
| every predicted table in gold | 103 | 110 | 110 | 108 |
| join keys equal given tables equal | 52 | 47 | 75 | 74 |
| correct given table set equal | 19 | 15 | 30 | 30 |
| correct given table set differs | 2 | 2 | 6 | 5 |
| exact plan | 4 | 4 | 9 | 2 |
| requests with same reference join | 33 | 23 | 0 | 0 |
| timeouts | 3 | 3 | 0 | 0 |
| EX-correct with a suspicious program | 5 | 4 | 9 | 9 |

| error category (requests; a request can have several) | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| projection | 146 | 148 | 136 | 129 |
| table grounding | 132 | 135 | 107 | 108 |
| predicate column | 118 | 113 | 102 | 102 |
| join key (different tables) | 98 | 106 | 94 | 88 |
| aggregation | 43 | 53 | 49 | 48 |
| ordering or LIMIT | 35 | 36 | 35 | 32 |
| DISTINCT | 28 | 29 | 30 | 31 |
| join: undeclared, non-gold key | 29 | 32 | 26 | 25 |
| grouping or HAVING | 25 | 32 | 27 | 25 |
| predicate literal | 21 | 24 | 29 | 25 |
| nested query involved | 22 | 23 | 24 | 24 |
| join: same-reference equality | 33 | 23 | 0 | 0 |
| join key (same tables) | 5 | 7 | 7 | 7 |
| no comparable plan | 3 | 3 | 3 | 3 |
| execution error | 2 | 0 | 4 | 1 |
| execution timeout | 3 | 3 | 0 | 0 |
| predicate operator or shape | 0 | 2 | 1 | 1 |
| predicate: same-reference equality | 2 | 0 | 0 | 0 |

| first divergence from the gold traversal | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| FROM table | 81 | 79 | 77 | 95 |
| join count or kind | 36 | 42 | 37 | 29 |
| joined table | 22 | 19 | 11 | 8 |
| ON column (operand 0) | 14 | 13 | 12 | 17 |
| gold not representable | 12 | 12 | 12 | 12 |
| WHERE predicate structure | 4 | 2 | 14 | 6 |
| WHERE column | 3 | 6 | 6 | 7 |
| exact | 4 | 4 | 9 | 2 |
| WHERE literal | 6 | 3 | 5 | 3 |
| SELECT column | 3 | 4 | 3 | 3 |
| FROM structure | 3 | 2 | 1 | 2 |
| SELECT aggregate | 1 | 2 | 1 | 4 |
| GROUP BY | 1 | 0 | 1 | 1 |
| LIMIT | 2 | 1 | 0 | 0 |
| ON column (operand 1) | 0 | 1 | 2 | 0 |
| DISTINCT | 0 | 0 | 1 | 1 |
| SELECT structure | 0 | 1 | 0 | 1 |
| query structure | 0 | 1 | 0 | 1 |

| teacher-forced decision (error rate) | steps per arm | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|---|
| SELECT other | 609 | 0.12 | 0.12 | 0.12 | 0.11 |
| WHERE operator | 432 | 0.26 | 0.31 | 0.32 | 0.30 |
| SELECT column | 249 | 0.42 | 0.39 | 0.33 | 0.34 |
| WHERE column | 246 | 0.36 | 0.39 | 0.29 | 0.30 |
| LIMIT | 240 | 0.03 | 0.03 | 0.03 | 0.03 |
| WHERE literal | 240 | 0.45 | 0.50 | 0.44 | 0.49 |
| HAVING operator | 204 | 0.00 | 0.00 | 0.00 | 0.00 |
| DISTINCT | 201 | 0.13 | 0.12 | 0.12 | 0.12 |
| GROUP BY other | 201 | 0.04 | 0.08 | 0.03 | 0.03 |
| QUERY | 201 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: no more tables | 198 | 0.22 | 0.20 | 0.18 | 0.18 |
| FROM table | 192 | 0.48 | 0.47 | 0.45 | 0.54 |
| STOP | 180 | 0.00 | 0.00 | 0.00 | 0.00 |
| ON column (operand 0) | 147 | 0.44 | 0.44 | 0.39 | 0.37 |
| ON column (operand 1) | 147 | 0.25 | 0.25 | 0.06 | 0.10 |
| ON operator | 147 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: add table | 147 | 0.58 | 0.58 | 0.43 | 0.46 |
| SELECT aggregate | 87 | 0.46 | 0.43 | 0.41 | 0.44 |
| ORDER BY column | 39 | 0.31 | 0.23 | 0.31 | 0.36 |
| GROUP BY column | 24 | 0.62 | 0.62 | 0.75 | 0.71 |
| ORDER BY other | 21 | 0.33 | 0.29 | 0.24 | 0.29 |
| FROM structure | 9 | 1.00 | 1.00 | 0.89 | 0.89 |
| HAVING column | 3 | 0.67 | 0.67 | 0.67 | 1.00 |
| HAVING literal | 3 | 0.33 | 1.00 | 0.00 | 0.67 |
| mean step accuracy (representable requests) | 180 | 0.786 | 0.782 | 0.809 | 0.801 |

