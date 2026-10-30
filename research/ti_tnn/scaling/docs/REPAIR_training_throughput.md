# C4/T4 training throughput: the repair before the pilot (development)

**Result.** A training update of 32 examples now takes 0.52 s on the DGX Spark, against 1.78 s in the smoke (3.4×
faster). The states are read from the encoding cache files, as the pilot will run, and matmuls use TF32. Training
also no longer needs every example's backbone states in memory. The pilot's training pool holds about 126 GB of BF16
states, more than this host's 121 GB of unified memory, so the old code could not have trained on it at all. Losses
match the smoke's training log to within 1.8 × 10⁻⁵ relative over the measured updates.

## What was slow, by example

In the smoke, an update takes the 32 examples of its slot in the sampling stream, in 4 micro-batches of 8. Each
example's states are a [T, 2560] matrix from the frozen 4B. T averages 5,211 tokens on the smoke questions, and the
longest is 26,544. Before the repair:
1. `prepare` widened the BF16 states to float32 at once, so each example held 4 bytes per value instead of 2.
2. `collate` allocated a float32 buffer on the CPU, padded to the longest context of the micro-batch, and copied
   every example into it. The batch was then copied to the GPU. One 26.5k-token example made its whole micro-batch
   about 2 GB.
3. The dense teacher-forcing tensors (legal-operator masks, span masks and so on) were rebuilt in Python for every
   update.

On the GPU an update took 1.78 s, about what it took on the CPU (2.2–2.4 s). So most of the time went to host work,
not to GPU arithmetic.

## What changed (`controller.py`, `train.py`)

| Change | Why it is exact, or what it changes |
|---|---|
| States stay in their cached dtype (BF16); `Trunk.memory` widens them on the GPU | The cache stores BF16, so widening on the device gives the same float32 values (tested) |
| Dense step tensors are built once per example, at its own sizes, and padded per batch | Equal to the old batch-size tensors (tested) |
| `collate` allocates the padded batch on the GPU and copies each example in once | Same values |
| Each update's 32 examples are grouped into micro-batches of similar length | Same examples, same summed loss and gradient; only the floating-point summation order changes |
| `prepare_examples(lazy=True)` keeps only each example's cache file; a background thread reads the next update's states while the current one computes | Same values (a file-backed run gives the same weight and optimizer digests as an in-memory one; tested) |
| A mapped file is read into ordinary memory on the prefetch thread (`clone()`) | Without it, copying a mapped tensor to the GPU read it page by page during the copy: 5.3 s per update |
| `TrainConfig.matmul_precision`: "highest" (float32, the smoke's) or "high" (TF32 tensor cores) | **Changes numerics.** Each product's inputs are rounded to a 10-bit mantissa. The BF16 states are exact in it; the weights are rounded. The setting is recorded in each run's config |

## Measurements

Timing runs (`cli train-timing`) used the smoke's 128 questions and recipe (T4, seed 100). The first 8–20 updates
were compared with the smoke's training log for the same updates and examples. Bundle: `evidence/repair_v1`, from
which `python -m scaling.evidence timing` rebuilds this table.

| States | Micro-batch | Matmuls | Seconds per update (after the first) | Prefetch wait s | Max relative loss difference | Peak GPU GB |
|---|---|---|---|---|---|---|
| (the smoke, before the repair) | 8 | float32 | 1.78 | — | — | — |
| in memory | 8 | float32 | 0.82 (20 updates) / 0.69 (8, profiled) | 0 | 1.4 × 10⁻⁶ | 17.1 |
| from files, before the `clone()` fix | 8 | float32 | 5.34 | 0.113 | 1.2 × 10⁻⁶ | 17.1 |
| from files | 8 | float32 | 0.65 (8, profiled) | 0.017 | 3.3 × 10⁻⁷ | 17.1 |
| **from files** | **8** | **TF32** | **0.52** | 0.013 | 1.7 × 10⁻⁵ | 17.1 |
| from files | 16 | TF32 | 0.80 | 0.015 | 1.8 × 10⁻⁵ | 24.9 |

**The profile** (3 updates, file-backed, float32): 0.53 s of GPU time per 0.65 s update, and the CPU spent most of
the rest waiting at synchronization points. About half the GPU time was float32 matrix products without tensor
cores. So the GPU's own work is now the floor. Larger micro-batches were slower, because they pad more.

**Disk.** The pilot's states cannot stay in the page cache, so most reads go to the NVMe drive. Each update needs
about 850 MB (32 × 26.6 MB on average). Direct reads measured 9.7–11.0 GB/s, and the update needs about 1.7 GB/s, so
the prefetch thread can keep ahead of training.

**Not measured:** c4_params (in the smoke its updates took the same time as T4's, 1.77 against 1.78 s); and a full
pilot-sized run reading cold files.

## After the review of cf1ad78

The reviewer confirmed every exactness claim above, and the prefetcher's thread safety and failure behaviour. A
missing or truncated file fails within 0.2 s, and no thread is left behind. Two gaps were found and fixed:

1. **A corrupted file of the right size trained silently.** Each new cache entry now records its states' sha256,
   checked whenever the entry is read (and so once per run at preparation). A file-backed example also records the
   CRC32 of the states it saw at preparation, and every later read is checked against it. CRC32 runs at 6.6 GB/s here,
   where sha256 runs at 2.3 GB/s; sha256 on every read would make the prefetch thread nearly as slow as training.
2. **A cache miss would have re-encoded silently.** The key includes the platform (GPU, torch, transformers,
   kernels), so any change between encoding and training turns every lookup into a miss. That would re-encode about
   2.6 GPU-hours of contexts. Now:
   - `require_hit` raises `CacheMiss` instead;
   - `Backbone.cache_reader` reads an identity's entries with no model, so the pilot can prepare on the CPU before
     reserving the GPU;
   - `evaluate` re-raises a cache miss rather than recording it as one request's abstention.

**Low findings, also fixed:**
- `collate` refuses a batch that mixes state dtypes;
- the summary vector is copied, so it no longer keeps a float32 example's whole state tensor alive;
- `evaluate` takes a `matmul_precision`, so the pilot can decode at its training precision.

## GPU accounting

Four `train-timing` reservations: 0.055 + 0.025 + 0.019 + 0.020 = **0.119 h**. The ledger total is now 2.979 h,
of which 2.861 h was the development study (`REPORT_dev.md`).

## What it changes for the pilot

| Item | Development report's estimate | Now |
|---|---|---|
| Train 4 runs × 1,500 updates | 3.0 h (1.78 s per update), or ≤ 0.5 h if the repair reached 0.3 s | **0.87 h** (0.52 s per update, TF32) |

The other items are unchanged. The draft pilot protocol (`PROTOCOL_pilot_draft.md`) carries the full budget.
