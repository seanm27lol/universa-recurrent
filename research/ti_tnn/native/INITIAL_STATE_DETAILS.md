# Initial-state study: all population metrics

These are aggregate tables from the preserved fixed-endpoint evaluations.
See [results and limits](INITIAL_STATE_RESULTS.md). A = `repeat_one`, B =
`multi_state`. Every population has 128 answers and 512 register positions.
Empty categories retain zero denominator and undefined accuracy in the local JSON.
No populations are pooled. Raw records, precise floating sums, inputs and weights
remain local under the aggregate identity cited in the results page.

## Complete strings, per-register accuracy, EOS and truncation

All `rj` columns are correct/128, including every malformed output as a failure.
The valid-only denominator for each register is the corresponding `Valid` count.
Malformed = 128 − Valid; correct-looking prefixes of malformed strings are not salvaged.

| Architecture | Condition | Population | Exact /128 | Valid /128 | EOS /128 | Truncated /128 | r0 /128 | r1 /128 | r2 /128 | r3 /128 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | A | train_anchor | 119 | 128 | 128 | 0 | 127 | 126 | 127 | 123 |
| dense | A | train_sample | 119 | 128 | 128 | 0 | 127 | 126 | 127 | 123 |
| dense | A | transition_values_id | 0 | 114 | 128 | 0 | 38 | 20 | 33 | 18 |
| dense | A | transition_values_ood | 0 | 117 | 128 | 0 | 0 | 0 | 1 | 0 |
| dense | A | program_values_id | 0 | 113 | 128 | 0 | 25 | 16 | 21 | 17 |
| dense | A | program_values_ood | 0 | 108 | 128 | 0 | 6 | 8 | 9 | 5 |
| dense | A | dev | 0 | 105 | 128 | 0 | 12 | 11 | 6 | 11 |
| dense | A | composition | 0 | 92 | 128 | 0 | 7 | 11 | 9 | 6 |
| dense | A | longer | 0 | 0 | 100 | 28 | 0 | 0 | 0 | 0 |
| dense | B | train_anchor | 2 | 128 | 128 | 0 | 41 | 49 | 49 | 52 |
| dense | B | train_sample | 2 | 128 | 128 | 0 | 46 | 50 | 51 | 49 |
| dense | B | transition_values_id | 2 | 128 | 128 | 0 | 55 | 33 | 51 | 50 |
| dense | B | transition_values_ood | 0 | 126 | 128 | 0 | 5 | 4 | 8 | 5 |
| dense | B | program_values_id | 2 | 128 | 128 | 0 | 38 | 36 | 46 | 33 |
| dense | B | program_values_ood | 0 | 125 | 128 | 0 | 21 | 21 | 22 | 22 |
| dense | B | dev | 1 | 126 | 128 | 0 | 22 | 21 | 29 | 24 |
| dense | B | composition | 0 | 101 | 128 | 0 | 15 | 16 | 16 | 13 |
| dense | B | longer | 0 | 0 | 127 | 1 | 0 | 0 | 0 | 0 |
| shared | A | train_anchor | 57 | 123 | 128 | 0 | 93 | 104 | 90 | 96 |
| shared | A | train_sample | 57 | 123 | 128 | 0 | 93 | 104 | 90 | 96 |
| shared | A | transition_values_id | 0 | 113 | 128 | 0 | 21 | 21 | 25 | 16 |
| shared | A | transition_values_ood | 0 | 70 | 128 | 0 | 1 | 5 | 5 | 0 |
| shared | A | program_values_id | 2 | 108 | 128 | 0 | 23 | 26 | 33 | 18 |
| shared | A | program_values_ood | 0 | 78 | 128 | 0 | 8 | 10 | 6 | 3 |
| shared | A | dev | 0 | 105 | 128 | 0 | 18 | 7 | 13 | 6 |
| shared | A | composition | 0 | 81 | 128 | 0 | 11 | 8 | 6 | 4 |
| shared | A | longer | 0 | 0 | 65 | 63 | 0 | 0 | 0 | 0 |
| shared | B | train_anchor | 1 | 128 | 128 | 0 | 39 | 38 | 33 | 39 |
| shared | B | train_sample | 1 | 127 | 128 | 0 | 35 | 46 | 37 | 35 |
| shared | B | transition_values_id | 0 | 128 | 128 | 0 | 33 | 23 | 35 | 34 |
| shared | B | transition_values_ood | 0 | 127 | 128 | 0 | 6 | 5 | 4 | 6 |
| shared | B | program_values_id | 3 | 128 | 128 | 0 | 37 | 37 | 46 | 32 |
| shared | B | program_values_ood | 1 | 128 | 128 | 0 | 23 | 22 | 19 | 20 |
| shared | B | dev | 0 | 126 | 128 | 0 | 20 | 25 | 23 | 27 |
| shared | B | composition | 0 | 98 | 128 | 0 | 15 | 14 | 9 | 9 |
| shared | B | longer | 0 | 0 | 81 | 47 | 0 | 0 | 0 | 0 |
| untyped | A | train_anchor | 43 | 126 | 128 | 0 | 98 | 106 | 85 | 87 |
| untyped | A | train_sample | 43 | 126 | 128 | 0 | 98 | 106 | 85 | 87 |
| untyped | A | transition_values_id | 0 | 110 | 128 | 0 | 19 | 19 | 12 | 17 |
| untyped | A | transition_values_ood | 0 | 90 | 128 | 0 | 0 | 0 | 0 | 1 |
| untyped | A | program_values_id | 0 | 98 | 128 | 0 | 17 | 10 | 11 | 15 |
| untyped | A | program_values_ood | 0 | 93 | 128 | 0 | 5 | 11 | 8 | 12 |
| untyped | A | dev | 0 | 96 | 128 | 0 | 8 | 10 | 8 | 10 |
| untyped | A | composition | 0 | 81 | 128 | 0 | 7 | 10 | 5 | 6 |
| untyped | A | longer | 0 | 0 | 53 | 75 | 0 | 0 | 0 | 0 |
| untyped | B | train_anchor | 0 | 126 | 128 | 0 | 37 | 40 | 32 | 27 |
| untyped | B | train_sample | 1 | 126 | 128 | 0 | 35 | 36 | 30 | 33 |
| untyped | B | transition_values_id | 0 | 126 | 128 | 0 | 40 | 34 | 28 | 22 |
| untyped | B | transition_values_ood | 0 | 128 | 128 | 0 | 4 | 4 | 4 | 2 |
| untyped | B | program_values_id | 1 | 123 | 128 | 0 | 33 | 25 | 30 | 26 |
| untyped | B | program_values_ood | 0 | 126 | 128 | 0 | 19 | 18 | 16 | 18 |
| untyped | B | dev | 0 | 120 | 127 | 1 | 18 | 16 | 18 | 17 |
| untyped | B | composition | 0 | 100 | 128 | 0 | 15 | 10 | 9 | 14 |
| untyped | B | longer | 0 | 0 | 58 | 70 | 0 | 0 | 0 | 0 |
| typed | A | train_anchor | 38 | 121 | 128 | 0 | 90 | 95 | 79 | 85 |
| typed | A | train_sample | 38 | 121 | 128 | 0 | 90 | 95 | 79 | 85 |
| typed | A | transition_values_id | 0 | 108 | 128 | 0 | 31 | 11 | 12 | 16 |
| typed | A | transition_values_ood | 0 | 76 | 128 | 0 | 3 | 1 | 1 | 1 |
| typed | A | program_values_id | 0 | 98 | 128 | 0 | 19 | 16 | 12 | 8 |
| typed | A | program_values_ood | 0 | 91 | 128 | 0 | 8 | 12 | 9 | 6 |
| typed | A | dev | 0 | 93 | 128 | 0 | 9 | 9 | 5 | 8 |
| typed | A | composition | 1 | 82 | 128 | 0 | 10 | 6 | 7 | 2 |
| typed | A | longer | 0 | 0 | 49 | 79 | 0 | 0 | 0 | 0 |
| typed | B | train_anchor | 0 | 126 | 128 | 0 | 32 | 33 | 33 | 26 |
| typed | B | train_sample | 1 | 127 | 128 | 0 | 27 | 35 | 32 | 37 |
| typed | B | transition_values_id | 0 | 125 | 128 | 0 | 37 | 21 | 38 | 22 |
| typed | B | transition_values_ood | 0 | 127 | 128 | 0 | 5 | 4 | 7 | 4 |
| typed | B | program_values_id | 2 | 127 | 128 | 0 | 34 | 31 | 45 | 31 |
| typed | B | program_values_ood | 0 | 128 | 128 | 0 | 14 | 21 | 18 | 22 |
| typed | B | dev | 1 | 125 | 128 | 0 | 18 | 16 | 21 | 15 |
| typed | B | composition | 0 | 100 | 128 | 0 | 11 | 7 | 11 | 13 |
| typed | B | longer | 0 | 0 | 26 | 102 | 0 | 0 | 0 | 0 |

## Changed and unchanged reference values

Changed means reference final value differs from that example's initial value;
unchanged includes both untouched registers and writes that preserve the value.
Each cell is correct/total positions. The groups sum to 512, with malformed
outputs contributing failures to their reference groups. A and B may have different
category denominators in `train_sample` because they use different initial vectors.
Per-register changed/unchanged counts and valid-only variants remain in local JSON.

| Architecture | Population | A changed | B changed | A unchanged | B unchanged |
|---|---|---:|---:|---:|---:|
| dense | train_anchor | 193/200 | 62/200 | 310/312 | 129/312 |
| dense | train_sample | 193/200 | 60/205 | 310/312 | 136/307 |
| dense | transition_values_id | 9/114 | 22/114 | 100/398 | 167/398 |
| dense | transition_values_ood | 1/113 | 22/113 | 0/399 | 0/399 |
| dense | program_values_id | 48/293 | 82/293 | 31/219 | 71/219 |
| dense | program_values_ood | 26/290 | 79/290 | 2/222 | 7/222 |
| dense | dev | 19/293 | 39/293 | 21/219 | 57/219 |
| dense | composition | 12/302 | 14/302 | 21/210 | 46/210 |
| dense | longer | 0/408 | 0/408 | 0/104 | 0/104 |
| shared | train_anchor | 146/200 | 51/200 | 237/312 | 98/312 |
| shared | train_sample | 146/200 | 60/205 | 237/312 | 93/307 |
| shared | transition_values_id | 24/114 | 23/114 | 59/398 | 102/398 |
| shared | transition_values_ood | 11/113 | 21/113 | 0/399 | 0/399 |
| shared | program_values_id | 61/293 | 94/293 | 39/219 | 58/219 |
| shared | program_values_ood | 27/290 | 81/290 | 0/222 | 3/222 |
| shared | dev | 21/293 | 45/293 | 23/219 | 50/219 |
| shared | composition | 13/302 | 10/302 | 16/210 | 37/210 |
| shared | longer | 0/408 | 0/408 | 0/104 | 0/104 |
| untyped | train_anchor | 146/200 | 44/200 | 230/312 | 92/312 |
| untyped | train_sample | 146/200 | 47/205 | 230/312 | 87/307 |
| untyped | transition_values_id | 13/114 | 16/114 | 54/398 | 108/398 |
| untyped | transition_values_ood | 1/113 | 14/113 | 0/399 | 0/399 |
| untyped | program_values_id | 33/293 | 65/293 | 20/219 | 49/219 |
| untyped | program_values_ood | 35/290 | 67/290 | 1/222 | 4/222 |
| untyped | dev | 15/293 | 30/293 | 21/219 | 39/219 |
| untyped | composition | 11/302 | 20/302 | 17/210 | 28/210 |
| untyped | longer | 0/408 | 0/408 | 0/104 | 0/104 |
| typed | train_anchor | 146/200 | 43/200 | 203/312 | 81/312 |
| typed | train_sample | 146/200 | 53/205 | 203/312 | 78/307 |
| typed | transition_values_id | 18/114 | 21/114 | 52/398 | 97/398 |
| typed | transition_values_ood | 6/113 | 20/113 | 0/399 | 0/399 |
| typed | program_values_id | 33/293 | 79/293 | 22/219 | 62/219 |
| typed | program_values_ood | 30/290 | 71/290 | 5/222 | 4/222 |
| typed | dev | 9/293 | 29/293 | 22/219 | 41/219 |
| typed | composition | 11/302 | 15/302 | 14/210 | 27/210 |
| typed | longer | 0/408 | 0/408 | 0/104 | 0/104 |

## Teacher-forced answer-component cross entropy

Each population covers all packed rows, with 2,176 answer+EOS labels = 512 numeric
value labels + 1,664 formatting/EOS labels. Values below are mean CE in nats/token,
rounded to six decimals; exact sums/counts remain saved. This is teacher-forced
loss, separate from free-generation accuracy and valid-only numerical error.

| Architecture | Population | A answer+EOS | B answer+EOS | A numeric | B numeric | A formatting/EOS | B formatting/EOS |
|---|---|---:|---:|---:|---:|---:|---:|
| dense | train_anchor | 0.083531 | 0.546815 | 0.340250 | 2.311782 | 0.004540 | 0.003749 |
| dense | train_sample | 0.083531 | 0.532032 | 0.340250 | 2.249029 | 0.004540 | 0.003725 |
| dense | transition_values_id | 0.930211 | 0.535048 | 3.869062 | 2.261547 | 0.025949 | 0.003818 |
| dense | transition_values_ood | 2.007589 | 1.219467 | 8.315922 | 5.171228 | 0.066564 | 0.003540 |
| dense | program_values_id | 1.035175 | 0.598123 | 4.195737 | 2.528120 | 0.062694 | 0.004277 |
| dense | program_values_ood | 1.555590 | 0.911668 | 6.442397 | 3.850743 | 0.051958 | 0.007338 |
| dense | dev | 1.211788 | 0.690013 | 4.895117 | 2.911943 | 0.078457 | 0.006342 |
| dense | composition | 1.516475 | 0.935657 | 5.021949 | 3.162102 | 0.437868 | 0.250597 |
| dense | longer | 3.884570 | 2.426209 | 5.302556 | 3.290402 | 3.448266 | 2.160303 |
| shared | train_anchor | 0.239760 | 0.590634 | 1.003034 | 2.496611 | 0.004906 | 0.004179 |
| shared | train_sample | 0.239760 | 0.575347 | 1.003034 | 2.431757 | 0.004906 | 0.004144 |
| shared | transition_values_id | 0.943524 | 0.585060 | 3.703462 | 2.472594 | 0.094312 | 0.004280 |
| shared | transition_values_ood | 2.067626 | 1.395586 | 8.559042 | 5.919138 | 0.070268 | 0.003723 |
| shared | program_values_id | 0.914154 | 0.622057 | 3.714833 | 2.625558 | 0.052406 | 0.005595 |
| shared | program_values_ood | 1.594325 | 1.021560 | 6.584270 | 4.328580 | 0.058957 | 0.004015 |
| shared | dev | 1.132603 | 0.723172 | 4.535444 | 3.043187 | 0.085575 | 0.009321 |
| shared | composition | 1.371251 | 1.004489 | 4.549416 | 3.202325 | 0.393354 | 0.328232 |
| shared | longer | 3.668455 | 2.486284 | 4.639508 | 3.347352 | 3.369669 | 2.221340 |
| untyped | train_anchor | 0.264756 | 0.642559 | 1.109804 | 2.718064 | 0.004742 | 0.003942 |
| untyped | train_sample | 0.264756 | 0.630326 | 1.109804 | 2.665912 | 0.004742 | 0.003992 |
| untyped | transition_values_id | 0.935558 | 0.645644 | 3.739508 | 2.729791 | 0.072804 | 0.004368 |
| untyped | transition_values_ood | 1.751503 | 1.387581 | 7.061605 | 5.880853 | 0.117625 | 0.005035 |
| untyped | program_values_id | 0.972029 | 0.674931 | 3.876568 | 2.840148 | 0.078325 | 0.008710 |
| untyped | program_values_ood | 1.392083 | 1.000883 | 5.634259 | 4.237321 | 0.086799 | 0.005055 |
| untyped | dev | 1.131542 | 0.757303 | 4.416950 | 3.147631 | 0.120648 | 0.021818 |
| untyped | composition | 1.454692 | 1.119660 | 4.394839 | 3.270806 | 0.550031 | 0.457768 |
| untyped | longer | 4.009693 | 3.262514 | 5.244145 | 3.345934 | 3.629862 | 3.236846 |
| typed | train_anchor | 0.297951 | 0.650282 | 1.251240 | 2.749990 | 0.004631 | 0.004218 |
| typed | train_sample | 0.297951 | 0.638764 | 1.251240 | 2.700835 | 0.004631 | 0.004280 |
| typed | transition_values_id | 0.977807 | 0.663024 | 3.784141 | 2.765962 | 0.114320 | 0.015966 |
| typed | transition_values_ood | 1.855233 | 1.324702 | 6.831395 | 5.617883 | 0.324107 | 0.003723 |
| typed | program_values_id | 0.972512 | 0.669788 | 3.802982 | 2.831350 | 0.101598 | 0.004693 |
| typed | program_values_ood | 1.237843 | 1.003610 | 4.972126 | 4.251667 | 0.088833 | 0.004208 |
| typed | dev | 1.118322 | 0.739849 | 4.230478 | 3.120631 | 0.160736 | 0.007301 |
| typed | composition | 1.609552 | 1.088830 | 4.325072 | 3.273090 | 0.774007 | 0.416751 |
| typed | longer | 4.171261 | 3.306380 | 5.205957 | 3.338519 | 3.852892 | 3.296491 |

## Supervised state readouts

Every packed row was evaluated. Cells show correct/count for each pass's supervised
register-state targets, with the existing before/after/after/after schedule.
Pass 2 reads the supervised after-state before the downstream pass-3 read.
Readout accuracy does not establish causal use of that state for generation.

| Architecture | Condition | Population | Pass 0 | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---:|---:|---:|---:|
| dense | A | train_anchor | 85/1152 | 275/1152 | 363/1152 | 405/1152 |
| dense | A | train_sample | 85/1152 | 275/1152 | 363/1152 | 405/1152 |
| dense | A | transition_values_id | 23/512 | 64/512 | 72/512 | 67/512 |
| dense | A | transition_values_ood | 0/512 | 31/512 | 32/512 | 28/512 |
| dense | A | program_values_id | 93/1792 | 254/1792 | 265/1792 | 293/1792 |
| dense | A | program_values_ood | 53/1792 | 144/1792 | 158/1792 | 141/1792 |
| dense | A | dev | 97/1800 | 227/1800 | 245/1800 | 245/1800 |
| dense | A | composition | 67/1652 | 154/1652 | 158/1652 | 155/1652 |
| dense | A | longer | 219/3540 | 485/3540 | 508/3540 | 479/3540 |
| dense | B | train_anchor | 86/1152 | 180/1152 | 205/1152 | 227/1152 |
| dense | B | train_sample | 69/1152 | 192/1152 | 219/1152 | 220/1152 |
| dense | B | transition_values_id | 24/512 | 63/512 | 75/512 | 77/512 |
| dense | B | transition_values_ood | 0/512 | 34/512 | 34/512 | 33/512 |
| dense | B | program_values_id | 97/1792 | 272/1792 | 305/1792 | 316/1792 |
| dense | B | program_values_ood | 51/1792 | 183/1792 | 206/1792 | 210/1792 |
| dense | B | dev | 119/1800 | 240/1800 | 256/1800 | 254/1800 |
| dense | B | composition | 93/1652 | 151/1652 | 169/1652 | 163/1652 |
| dense | B | longer | 277/3540 | 521/3540 | 536/3540 | 526/3540 |
| shared | A | train_anchor | 109/1152 | 291/1152 | 443/1152 | 469/1152 |
| shared | A | train_sample | 109/1152 | 291/1152 | 443/1152 | 469/1152 |
| shared | A | transition_values_id | 24/512 | 73/512 | 81/512 | 83/512 |
| shared | A | transition_values_ood | 0/512 | 26/512 | 26/512 | 25/512 |
| shared | A | program_values_id | 121/1792 | 255/1792 | 299/1792 | 293/1792 |
| shared | A | program_values_ood | 66/1792 | 145/1792 | 159/1792 | 158/1792 |
| shared | A | dev | 105/1800 | 215/1800 | 264/1800 | 256/1800 |
| shared | A | composition | 72/1652 | 147/1652 | 178/1652 | 187/1652 |
| shared | A | longer | 269/3540 | 432/3540 | 445/3540 | 454/3540 |
| shared | B | train_anchor | 84/1152 | 161/1152 | 232/1152 | 244/1152 |
| shared | B | train_sample | 68/1152 | 170/1152 | 224/1152 | 231/1152 |
| shared | B | transition_values_id | 30/512 | 65/512 | 86/512 | 94/512 |
| shared | B | transition_values_ood | 0/512 | 29/512 | 35/512 | 36/512 |
| shared | B | program_values_id | 114/1792 | 246/1792 | 316/1792 | 328/1792 |
| shared | B | program_values_ood | 49/1792 | 163/1792 | 205/1792 | 212/1792 |
| shared | B | dev | 136/1800 | 214/1800 | 272/1800 | 280/1800 |
| shared | B | composition | 109/1652 | 127/1652 | 172/1652 | 172/1652 |
| shared | B | longer | 295/3540 | 446/3540 | 497/3540 | 489/3540 |
| untyped | A | train_anchor | 121/1152 | 354/1152 | 462/1152 | 511/1152 |
| untyped | A | train_sample | 121/1152 | 354/1152 | 462/1152 | 511/1152 |
| untyped | A | transition_values_id | 21/512 | 72/512 | 77/512 | 78/512 |
| untyped | A | transition_values_ood | 1/512 | 34/512 | 36/512 | 34/512 |
| untyped | A | program_values_id | 107/1792 | 255/1792 | 283/1792 | 259/1792 |
| untyped | A | program_values_ood | 54/1792 | 161/1792 | 159/1792 | 154/1792 |
| untyped | A | dev | 84/1800 | 218/1800 | 217/1800 | 232/1800 |
| untyped | A | composition | 75/1652 | 159/1652 | 161/1652 | 166/1652 |
| untyped | A | longer | 207/3540 | 433/3540 | 428/3540 | 404/3540 |
| untyped | B | train_anchor | 82/1152 | 149/1152 | 177/1152 | 184/1152 |
| untyped | B | train_sample | 73/1152 | 170/1152 | 182/1152 | 198/1152 |
| untyped | B | transition_values_id | 24/512 | 57/512 | 65/512 | 68/512 |
| untyped | B | transition_values_ood | 1/512 | 35/512 | 37/512 | 36/512 |
| untyped | B | program_values_id | 117/1792 | 255/1792 | 293/1792 | 294/1792 |
| untyped | B | program_values_ood | 64/1792 | 183/1792 | 203/1792 | 215/1792 |
| untyped | B | dev | 113/1800 | 198/1800 | 204/1800 | 204/1800 |
| untyped | B | composition | 95/1652 | 133/1652 | 121/1652 | 117/1652 |
| untyped | B | longer | 266/3540 | 482/3540 | 467/3540 | 458/3540 |
| typed | A | train_anchor | 207/1152 | 595/1152 | 752/1152 | 814/1152 |
| typed | A | train_sample | 207/1152 | 595/1152 | 752/1152 | 814/1152 |
| typed | A | transition_values_id | 49/512 | 107/512 | 110/512 | 121/512 |
| typed | A | transition_values_ood | 1/512 | 31/512 | 24/512 | 23/512 |
| typed | A | program_values_id | 140/1792 | 332/1792 | 336/1792 | 338/1792 |
| typed | A | program_values_ood | 72/1792 | 162/1792 | 158/1792 | 164/1792 |
| typed | A | dev | 126/1800 | 296/1800 | 307/1800 | 300/1800 |
| typed | A | composition | 103/1652 | 187/1652 | 204/1652 | 219/1652 |
| typed | A | longer | 266/3540 | 559/3540 | 542/3540 | 499/3540 |
| typed | B | train_anchor | 110/1152 | 232/1152 | 254/1152 | 266/1152 |
| typed | B | train_sample | 86/1152 | 234/1152 | 267/1152 | 281/1152 |
| typed | B | transition_values_id | 38/512 | 93/512 | 105/512 | 103/512 |
| typed | B | transition_values_ood | 1/512 | 37/512 | 39/512 | 36/512 |
| typed | B | program_values_id | 139/1792 | 332/1792 | 371/1792 | 389/1792 |
| typed | B | program_values_ood | 67/1792 | 194/1792 | 210/1792 | 215/1792 |
| typed | B | dev | 133/1800 | 262/1800 | 289/1800 | 289/1800 |
| typed | B | composition | 113/1652 | 193/1652 | 227/1652 | 227/1652 |
| typed | B | longer | 271/3540 | 560/3540 | 571/3540 | 578/3540 |

## Final checkpoint identities and per-arm training timers

All checkpoints are the registered final update 1,424. All cursors are 11,376;
all exposure counts are 2,590,592; no replayed updates occurred. Timers below
include data/forward/backward/optimizer work only, not total reservation cost.
The result page reports the complete reservation including CPU evaluations.

| Architecture | Condition | Final SHA256 | Recorded training seconds |
|---|---|---|---:|
| dense | A | `722883eb495178af3bb6082510095e0cebf3a43b5215cb36eb04e9fd04bd53ec` | 86.392035 |
| dense | B | `80622b5f93a93cfc9a75823880aab880aa97a9e17e005e728aebf417c48584ed` | 91.386760 |
| shared | A | `e4fe4ab988403d36bf6915543c2a1f840cd514fabf9724d9fc6ef4cc36d7dc4b` | 83.775792 |
| shared | B | `025c9e8553acf82a4830417469451546fa3ccc1cc02478a9ad43cda0502e1016` | 91.093312 |
| untyped | A | `c6c8b179e77d79bc1f7b6aa8723ddec10775d18be572fc7e48e2d461425657a4` | 94.272697 |
| untyped | B | `e3af72bcfa87463ebb80aaa9942d7ef15059b34ecc597c2859b9c71041f4c44b` | 99.998568 |
| typed | A | `5dc90ef1de75d77c4dcafa1a056b0eecad03a18df2aa9e7f9f287d9f5ac1ceaf` | 103.183011 |
| typed | B | `4abbae50d4d56a318660479761885e7a7b8af2c36a67a49583b6298335c3f6a2` | 103.776670 |
