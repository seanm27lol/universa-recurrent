Reporting-correction verification (separate from the original reconstruction in ../, which is unchanged).

What was corrected: the public reconstruction (dbe_report.py). The run, its records, the frozen harness, the
analysis and the original reconstruction did not change. Owner's requirements:
- comparison coverage: the analysis's exact key set, the frozen protocol and population binding, a stable run
  identity, and gpu_reservations and the ledger recomputed from the bundled ledger;
- raw-value validation with #73's protections;
- rejection tests with regenerated checksums.

round1/  the independent reviewer's verification of d304831:
         - 195 bundle-corruption probes and 5 pin probes;
         - 1 case wrongly accepted: a run directory renamed consistently with analysis.json and the analysis
           manifest;
         - the relocated reconstruction byte-identical (sha256 b81cbe67...);
         - the earlier review's outputs unchanged.
round2/  the reviewer's confirmation of d8c48d7, after the run directory was bound to the ledger's start entry:
         - 198 probes, none wrongly accepted, all 178 earlier refusals kept, 1e-12 round-off accepted;
         - 5 of 5 pin probes refused;
         - the relocated reconstruction byte-identical;
         - 138 focused tests passed.

Remaining judgement calls, accepted and changing no displayed value:
- a full relabel that also rewrites the bundled ledger (the bundle is then self-consistent);
- the same other commit in both manifests;
- the ledger end entry's informational hours and a heartbeat below the maximum;
- summary.json's reservation hours (about 5e-8 h from the ledger by construction) and load_seconds;
- -0.0 and 1e308 in two undisplayed times.

Files are copied byte for byte from the reviewer's output directories and may contain machine paths.
