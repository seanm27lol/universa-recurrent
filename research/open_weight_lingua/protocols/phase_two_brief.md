# Phase Two: Can a Language Description Preserve and Edit an Open Model's Activation?

**Codex research and implementation brief · 20 September 2026**

**Repository:** `seanm27lol/universa-recurrent`

**Status:** proposed experiment and implementation specification, not a completed result.

**First target:** `Qwen/Qwen2.5-7B-Instruct` with its released Natural Language Autoencoder (NLA) pair.
**Stopping rule:** one bounded pilot, then at most one locked validation. Preserve negative findings and close this phase rather than automatically adding experiments.

> Keep the project penetrable for anyone curious to understand it, while grounding its ideas in familiar examples and established fields.

## 0. Codex: start here

Read this brief in full and the repository's applicable `AGENTS.md` files. Then implement **Milestone 1 in Section 11**, not a new architecture or training pipeline. Produce working code, tests, a compatibility report, and a runnable small real-model smoke-test command. Do not stop at another proposal.

If GPU/model access is unavailable in your execution environment, complete the CPU-testable work and supply the exact DGX command. Label the real-model checks **not run**. Do not fabricate model results, downloaded files, successful remote writes, or passed GPU tests.

Keep Phase One's runtime, dependencies, historical scripts, protocols, checkpoints, reports, and conclusions unchanged. The repository HEAD verified while preparing this brief was `ecc8d79873b6f76b4e896986092c7ce8909cdf75`; inspect the current HEAD and diff before editing. Do not reset a newer checkout to that commit. Work on an isolated branch/worktree where the execution environment permits it. Do not auto-merge, change visibility, or publish weights or raw activation data.

**Desired first handoff:** a small reviewed patch that can read one real Qwen activation, restore it without changing behavior, call the released describer/reconstructor correctly, and report the resulting behavioral difference. This is an implementation gate, not the scientific conclusion.

---

## 1. Concrete question and relation to Phase One

Imagine a model reading:

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The correct answer is 5. We inspect one internal activation before the model answers. A separate model describes that activation in language. Another reconstructs an activation from the description. We replace the original activation and let the remaining computation run.

**Does the reconstructed activation preserve the answer distribution? If the description explicitly represents the current value of x, does a minimal edit to that statement produce a specific, reproducible behavioral change?**

```text
                         Frozen Qwen target
                                |
                One specified residual-stream vector
                                |
                 Released activation verbalizer (AV)
                                |
                       Language description
                                |
              Released activation reconstructor (AR)
                                |
             Reconstructed direction + declared norm
                                |
              Replace ONE vector in the original run
                                |
          Compare behavior, targeted changes, and costs
```

### What transfers, and what does not

Phase One supported approximate numerical state preservation and editing of **known mathematical fields**. Its final local linear predictor reduced RMS response-prediction error by about 30.9% versus predicting no change, but failed the much stricter close-prediction criterion. Those findings are specific to the structured synthetic model, not Qwen. The published closeout remains the source of record. [S1]

Here we transfer the **preserve → intervene → continue → measure** methodology. We do not transfer cycle bases, structural certificates, accuracy claims, or purported neuron meanings. Qwen's hidden coordinates are learned, not named quantities whose semantics we already know.

Qwen is a conventional transformer target. This phase tests language-mediated activation reconstruction and intervention **across the remaining layers**, not adaptive recurrent depth. A recurrent open-weight target or learned choice of reasoning spaces requires a separate later proposal. It is outside the present stopping rule. [S2]

---

## 2. Verified starting points and unresolved implementation details

The following are **source-derived facts**, distinct from the experimental choices proposed below.

| Component | Source-derived starting point |
|---|---|
| Target | `Qwen/Qwen2.5-7B-Instruct`: the model card lists 7.61B parameters, 28 layers, BF16 tensors, and Apache-2.0 licensing. [S2] |
| Describer | `kitft/nla-qwen2.5-7b-L20-av`: released activation verbalizer paired with the AR checkpoint below. [S3] |
| Reconstructor | `kitft/nla-qwen2.5-7b-L20-ar`: released activation reconstructor for the same target and extraction site. [S4] |
| Extraction site | The released pair targets the residual-stream output of block index 20. Confirm the zero-based indexing against the pinned code; do not infer it from “two-thirds depth.” [S3–S6] |
| Geometry | The released recipe evaluates normalized directions. A directional reconstruction score does not establish recovered activation magnitude or preserved target behavior. [S5–S8] |
| Inference code | `kitft/nla-inference` provides the lightweight recipe and `NLAClient`/`NLACritic` interfaces. Use the inspected pinned implementation rather than inventing their signatures. [S5–S6] |
| Scientific precedent | The NLA authors already describe language explanations, reconstruction, edited explanations, and steering using reconstructed differences. Do not present those ideas as inventions of this project. [S8] |
| Hardware | NVIDIA documents DGX Spark's Arm CPU and 128 GB unified system memory. That makes a short-context pilot plausible, not a guarantee of package compatibility or speed. [S10] |

**Verification status:** model cards, repository documentation, and the current Phase-One repository were inspected for this brief. No Qwen/AV/AR weights were loaded and no DGX compatibility test was performed here. Exact weight revisions, metadata values, tokenizer behavior, and backend compatibility must be resolved by the preflight.

The NLA paper reports confabulations and layer sensitivity. Plausible wording and good activation reconstruction are therefore hypotheses to test against behavior, not sufficient semantic evidence. [S8]

### Required preflight lock

Resolve and record immutable revisions for the target, AV, AR, tokenizer files, metadata sidecars, and inference source. Record their licenses/attributions. Download only those pinned revisions when the user runs the fetch step. Do not silently follow `main` on a later run.

Load each checkpoint's `nla_meta.yaml`. Assert the target width and layer, injection marker/token and its context, exact AV/AR prompt templates, scale conventions, AR depth, and value-head dimensions against the actual tokenizer/configuration. Missing or contradictory metadata is a stop condition, not permission to guess.

The AR is **not** an ordinary text-generation call. The documented recipe uses a truncated backbone, removes final normalization from the reconstruction path, and loads a separate learned value head. Reproduce the pinned recipe, including its terminal-token convention. Do not substitute the target's language-model head. [S6]

---

## 3. Scope: three questions, not a claim of universal interpretation

| Question | Evidence sought | What a positive result would not show |
|---|---|---|
| **Preservation** | Reconstructed activations preserve measured behavior better than irrelevant descriptions and within a declared accuracy-loss margin. | Exact state recovery, complete reasoning-state compression, or human-semantic fidelity. |
| **Specificity** | A minimal edit to an actually present statement shifts the intended answer more than a no-edit or wrong-variable control. | Discovery of a uniquely localized variable, a complete causal account, or arbitrary English control. |
| **Cost** | Explicit bytes, model calls, time, and memory for the full procedure. | A speedup merely because the description is short. |

No fine-tuning of target, AV, or AR is planned. Fitting a numerical baseline on a separate calibration split is allowed and must be counted. No external LLM judge, paid API, safety-bypass experiment, private RF-MoE code, or new model family is needed.

**Attribution:** the AV and AR are external pretrained baselines. Our contribution in this phase would be the controlled transfer experiment, its implementation, and its evidence, including failures. Check prior work before making any novelty claim. [S8–S9]

---

## 4. The activation boundary must be correct

### Primary site

Use the output of the NLA-compatible block and the **last non-padding token of the fully tokenized target prompt, including the assistant-generation prefix**. Begin with batch size 1. Save token IDs, decoded token, absolute position, mask, layer index, tensor width/dtype, and hook path for every sample.

This token may encode response-format information more strongly than the variable of interest. That is a possible negative result. Do not search many tokens/layers until an attractive result appears. The pilot's donor and perturbation controls determine whether the selected site supports the intended assay.

Do not rely on a hardcoded `hidden_states[20]`: frameworks may count embeddings separately. Confirm the block-output hook against the corresponding hidden-state tensor for the pinned target implementation. Preserve tuple/model-output structure and all unmodified token vectors. Hook removal must be exception-safe.

### Identity checks

Before using any reconstruction:

1. Repeat unmodified inference with fixed inputs, dtype, backend, and seed.
2. Install a no-op hook; confirm no output change.
3. Capture and reinsert the same native-dtype vector; require exact tensor equality at the hook, identical discrete outputs, and the frozen numerical logit tolerance.
4. Confirm only the intended batch row/token/layer was changed. Run two different inputs back-to-back to detect stale hooks or state reuse.

Start with `atol=1e-5`, `rtol=1e-5` for logit comparison and exact greedy-answer agreement. If repeatability itself prevents this, investigate on smoke-test data and document a justified bound before the pilot. Never relax tolerances after inspecting validation results. Bitwise equality is preferable when the same execution path supports it; do not promise cross-device bitwise reproducibility.

### Caching and continuation

For the initial target tests, use `eval()`, inference mode, and `use_cache=False`. A fresh full forward pass with an output hook is acceptable: it still changes a precisely defined internal computation, although it recomputes the prefix and is **not a faster pause/resume engine**.

For autoregressive answers or teacher-forced scoring, preserve the same original prefix intervention on every recomputed pass. Do not drift the hook to the newest token or patch every position. For teacher forcing, capture the activation before any answer suffix is appended, and check that the prefix activation is causally unchanged by appended candidate answers.

All other token activations and the original prompt remain available. A single-vector patch is **not** replacement of the model's entire state, and absence of an effect can reflect distributed or redundant information. [S9]

---

## 5. Handle magnitude and information channels honestly

Let `h` be the target's native activation converted to float32 for measuring its norm, `n = ||h||₂`, `c = AV(h)`, and `r = AR(c)`. The proposed primary reconstruction is:

```text
h_reconstructed = n * r / ||r||₂
```

Compute normalization in float32, reject nonfinite or near-zero reconstructed norms, then cast once to the target's original dtype for patching. AV injection uses its **own metadata-defined normalization and scale**; that scale is not the norm used to reinstall the activation. [S5–S7]

**Call this text-plus-norm reconstruction.** The original norm is a retained per-example side channel. Count its four bytes, the patch-site identifiers, all shared metadata, and any other retained information. Do not say the decoder recovered the complete vector from text alone.

The decoder receives description text and fixed metadata, not the target prompt, source activation, correct answer, donor activation, or future states. Norm restoration happens in a separate explicit patching step. Add a fixed calibration-median-norm diagnostic on smoke/pilot data to expose reliance on the norm; do not select a norm policy from validation outcomes.

For direction diagnostics, use cosine similarity and `||h/||h|| - r/||r||||₂²`. For unit vectors, the latter equals `2(1-cosine)`. An **elementwise mean** additionally divides by dimension unless the vectors were scaled by `sqrt(d)`. Name the metric correctly; do not conflate the released normalized score with raw activation MSE. [S6]

The checkpoint/model-card reconstruction scores are not scores on our task and must not be copied into our results.

---

## 6. A small, paired task with exact answers

### Generator

Use short straight-line programs with two variables, initialization, assignment/copy, and addition/subtraction by constants. Use a small explicit interpreter, not `eval` or arbitrary code execution. Keep reference values in a bounded range such as 0–19, use 3–6 statements, and balance which variable is queried and which input receives the counterfactual edit.

Each **problem group** contains source program A, counterfactual B differing in one assignment, and questions about both the affected and unaffected variable. For example:

```text
A: x = 3; y = 8; x = x + 2       B: x = 4; y = 8; x = x + 2
Query x: expected 5              Query x: expected 6
Query y: expected 8              Query y: expected 8
```

Check in the generator that the intended output changes and the control output does not. Source/donor prompts must have aligned patch positions; either enforce equal tokenized length with an input-only rule or record an explicit alignment policy. Do not filter on a model's answer to manufacture a responsive dataset.

These facts give ground truth for program behavior, **not ground truth for the semantic contents of an activation**.

### Proposed budget and splits

| Split | Problem groups | Permitted use |
|---|---:|---|
| Engineering smoke | 8 | Correctness, backend/tokenizer checks, throughput estimate. |
| Numerical calibration | 256 | Fit numeric baseline and median norm. No neural fine-tuning. |
| Pilot | 128 | Assess feasibility, edit coverage, and finalize predeclared operational choices. |
| Locked validation | 512, as two 256-group blocks | Final evaluation after the protocol and source hashes are locked. |

These are proposed design counts, not completed runs. All paired prompts/questions from one group stay in the same split. Assign disjoint split namespaces/seeds, deduplicate canonical programs and tokenized prompts, and prevent overlap with smoke/calibration/pilot. Validate semantic-pair integrity before model evaluation. Tokenization rejections and repeated templates must be reported.

Count groups separately from prompt variants, extraction sites, controls, and repeats. A program appearing in four variants does not create four independent samples.

Freeze one target prompt template. Use a single description per activation, proposed greedy AV decoding with a 200-new-token ceiling. This is our deterministic experimental choice, not a replication of the authors' sampled-description evaluation. Missing closing tags or truncation are reported failures; no answer-guided retries, best-of-N selection, or manual rewriting of generated descriptions.

---

## 7. Required conditions and the edit-coverage gate

### Preservation conditions

| ID | Intervention | Purpose |
|---|---|---|
| P0 | Unmodified target | Behavioral reference. |
| P1 | Reinsert original activation | Mandatory adapter correctness gate. |
| P2 | Own NLA description → AR direction + original norm | Main reconstruction condition. |
| P3 | Another group's description → AR direction + receiver norm | Detect whether the specific description matters. Use a fixed input-independent derangement. |
| P4 | Calibration-fitted numerical reconstruction | Compare language with generic representation, not just corruption. |
| P5 | Norm-matched random direction | Coarse sensitivity diagnostic; not a meaningful semantic alternative. |

For P4, use a clearly specified PCA baseline on unit directions fitted only to calibration activations. Quantize coefficients to float16, normalize the reconstructed direction, then restore the same receiver norm. Give it no task labels. Choose its retained rank from a **declared byte-budget rule**, not per-example reconstruction quality. For example, allow at most the UTF-8 description's value bytes for coefficients: `rank = min(fitted_rank, floor(text_bytes/2))`.

The mean/basis, model weights, and metadata have shared costs. Report the actual rank, unused budget, and amortized/shared bytes. When the fitted rank caps storage below the language budget, call the condition **no-more-than-budget**, not exactly matched or an optimal compression baseline. Fit on the pooled calibration split only; selecting separate bases from true task labels is not allowed.

### Targeted edits: secondary and explicitly conditional

A frozen deterministic parser may edit a description only when it contains one unambiguous statement of a variable's **current** value. A quoted initial assignment is not necessarily a current value. Preserve all other wording; never insert a missing claim using the reference program's answer.

The editor receives only `(description, requested variable, requested new value)` and the frozen editing rule. The requested value is the intervention, not hidden evidence that the original description is true. An independent evaluator records whether the original explicit statement agrees with the reference program. Ambiguous, absent, or inconsistent descriptions receive a status, not a fabricated replacement.

Compare the edited reconstruction with the unedited reconstruction, a same-sized wrong-variable edit when eligible, and a **norm-matched raw donor direction** from counterfactual B. A donor patch keeps the receiver's remaining context; it need not reproduce the entire counterfactual prompt's behavior. It is a sensitivity/contrast control, not an oracle proof of localized semantics.

Also evaluate source/counterfactual questions about the unaffected variable. Interpret that as specificity across paired query contexts, not proof that every unrelated behavior is unchanged.

Report editing coverage over **all groups**, conditional outcomes on eligible groups, and the distribution of exclusion reasons. Proposed feasibility floor: at least 32 pilot groups permit the frozen edit rule. If not, report that this task/site did not provide a usable explicit-variable interface; do not turn a manual ground-truth description into “discovered semantics.” An explicitly labeled oracle-description sanity test may be shown on smoke examples only, outside scientific success counts.

No steering-strength sweep, learned editor, extra checkpoint, or layer search is part of this pilot. The NLA authors' reconstructed-difference steering is related work, not an untested fallback to adopt mid-validation. [S8]

---

## 8. Measurements and proposed decision rules

### Record behavior before interpretation

Use an exact answer scorer and greedy generation, plus full-sequence log probabilities for the source and counterfactual answers. Do not assume integer answers are one token. Freeze canonical answer formatting and suffix-tokenization before the pilot; record invalid formatting as an error. During sequence scoring, patch the original prompt position in every candidate pass without feeding the answer into AV or AR.

For preservation, report paired exact-answer loss versus P0, original-answer agreement, full-vocabulary next-token KL computed with stable float32/log-softmax arithmetic, answer log probabilities, cosine similarity, norm error, and failure rate. Answer agreement alone can be uninformative when the model is insensitive to the patch.

For editing, a useful continuous measure is:

```text
L = log P(counterfactual answer | receiver prompt, patch)
  - log P(original answer | receiver prompt, patch)
Effect of text edit = L(edited reconstruction) - L(unedited reconstruction)
```

Report that paired effect, counterfactual exact answers, wrong-variable differences, unaffected-variable errors, and raw-donor sensitivity. Near-zero donor effects invalidate ratio-style “fraction recovered” metrics; report absolute effects rather than divide by a tiny denominator.

### Proposed practical standards, to freeze before validation

These thresholds are **design choices**, not claims from a paper or universal definitions of interpretability.

| Decision | Proposed rule |
|---|---|
| Implementation correctness | P1 preserves the captured native tensor, discrete outputs, and fixed numerical tolerances. A failed gate blocks interpretation. |
| Task usable in pilot | Unmodified exact-answer accuracy at least 80% and donor/perturbation controls show that this site can affect the intended measurement. Otherwise stop with an assay limitation. |
| Limited behavioral preservation | One-sided 95% upper estimate of P2 accuracy loss versus P0 is at most 5 percentage points; P2 also beats P3 on correct-answer log probability with a positive lower confidence estimate. Require the pooled criteria and both validation-block point estimates in the intended direction. |
| Specific edit evidence, secondary | On prespecified eligible groups, text edits increase counterfactual preference versus no edit and wrong-variable controls; report paired uncertainty, coverage, and unaffected-variable losses. Do not label success from a few examples. |
| Efficiency | Descriptive complete costs only. No speedup claim without a separately justified matched-behavior timing comparison. |

Finalize numerical editing criteria in the pilot decision document before opening validation. If coverage or sensitivity is inadequate, do not run a nominal semantic-validation stage; close with preservation-only results and explicitly mark the edit hypothesis untested or unsupported. Never weaken a primary threshold on held-out results.

Bootstrap **whole problem groups**, preserving paired prompt variants and conditions. Report both validation blocks, pooled paired estimates, and uncertainty conditional on this single checkpoint, task, and site. Use a fixed seed and 3,000 group-resamples as the default. Repeated descriptions, hardware runs, and token positions are not independent models.

All failures need an intention-to-test accounting. Missing generations count against accuracy; exclude neither errors nor outliers silently. Do not manufacture finite KL values for failed reconstructions: report valid-case distributional metrics alongside the total failure denominator. If failures invalidate a claimed aggregate, mark that outcome inconclusive/failed.

---

## 9. DGX and software constraints

Use a separate `.venv-phase2` or pinned container. Do not upgrade Phase One's PyTorch/CUDA environment, global drivers, or root package version to satisfy Phase Two. Qwen, AV, and AR must retain their native released precision for the first scientific comparison; changing weight quantization is a separate intervention.

Start with short prompts, batch size 1, and sequential stages: target extraction, AV generation, AR reconstruction, target patching. Unload models between stages if necessary. Unified CPU/GPU memory is one shared pool; moving tensors to CPU does not create a second independent 128 GB budget. [S10]

Record available memory/storage before fetching, actual downloaded bytes, backend versions, GPU capability, device/dtype placement, and smoke throughput. The reviewed AV/AR file listings are approximately 15.2 GB and 10.9 GB, respectively; the target and caches are additional. Verify pinned totals instead of treating those figures as a runtime-memory estimate. [S3–S4]

The documented AV path uses SGLang embedding injection. It requires guarding against prefix-cache aliasing for distinct embeddings; follow and test the pinned inference recipe, including its radix-cache restriction. Do not assume every current SGLang/FlashAttention wheel supports the Spark's Arm/Blackwell stack. [S5]

A local Transformers AV implementation is acceptable only as an explicit alternative adapter reproducing the published embedding injection and prompt conventions. Test it on fixtures and real smoke examples. Do not silently replace the pretrained pair with generic prompting. The target hook path can use ordinary eager PyTorch independently of the AV backend.

Bind any local server to loopback, never expose an unauthenticated inference service. Avoid `trust_remote_code=True` unless the exact required code is inspected and pinned. Prefer safe tensor formats; do not load untrusted pickle/object archives. Keep all evaluation synthetic and local.

Before the pilot and validation, estimate wall time from measured smoke throughput. The proposed unattended budget is **eight hours per scientific stage**, not a promise of duration. If projected work exceeds it, report the estimate and stop for a scope decision before opening that stage. Do not shrink samples or conditions midway through a run.

---

## 10. Proposed repository layout and interfaces

These paths and commands are **to be implemented**. They are not claimed to exist now.

```text
research/open_weight_lingua/
  README.md                 # concrete example, claims and limits
  pyproject.toml            # isolated dependencies and entry points
  configs/pilot.yaml        # draft until pilot choices are recorded
  src/open_weight_lingua/
    preflight.py            # revisions, licenses, metadata, backend checks
    target.py               # capture and one-site patching
    nla_adapter.py          # published AV/AR recipe, explicit norm boundary
    tasks.py                # reference interpreter and grouped paired data
    controls.py             # numerical, shuffled, donor and perturbation controls
    text_edits.py           # frozen parser and honest coverage statuses
    metrics.py              # independently testable scoring
    runner.py               # smoke/pilot/validation stages
    audit.py                # recompute from saved evidence, no neural calls
  tests/
  scripts/                  # one-command stage runners, no hidden global setup
  protocols/                # pilot decision + immutable validation manifest
  reports/                  # curated summaries only; raw run directories ignored
```

Core contracts should make leakage difficult:

```python
capture(prompt_ids, attention_mask, site) -> ActivationRecord
verbalize(vector, fixed_nla_metadata, sampling_config) -> DescriptionRecord
reconstruct(description, fixed_nla_metadata) -> DirectionRecord
restore_norm(direction, explicitly_retained_norm) -> ReplacementVector
patch_and_score(original_prompt, site, replacement, scoring_config) -> BehaviorRecord
edit_description(description, requested_edit, frozen_edit_rule) -> EditResult
```

AV may see only its injected activation and fixed instruction template. AR may see only the description and fixed reconstruction template. Neither may receive task truth. No response-rule fitting or the Phase-One six-dimensional Hessian code should be transplanted into a 3,584-dimensional activation experiment.

### Evidence to save

Write an immutable manifest before inference, including protocol, split/group identities, input IDs, model/code/tokenizer revisions, sidecar hashes, backend flags, seeds, native dtype, extraction site, norm policy, prompt templates, description budget, baseline fit identity, controls, and metrics.

Save original and reconstructed selected vectors in safe numeric files, full descriptions, edited descriptions and diffs, actual norms, answer tokenizations, answer scores, next-token divergence statistics, generation outputs, failures, stage times, and counted bytes. Keep enough data to recompute headline outcomes without rerunning AV/AR. Full-vocabulary distributions may be stored separately; disclose when an upload omits them rather than claiming an independent KL replay.

Use fresh run directories; never overwrite or silently reuse incompatible artifacts. Resumption must verify the same locked manifest and all completed per-example identities. Emit machine-readable completion status and an aggregate Markdown report with no cherry-picked examples. Count successful, failed, skipped, and uneditable groups separately.

Model weights, credentials, raw activations, and user-local paths must not be committed to the public repository. Prepare a reports-only ZIP; include a clear inventory of evidence retained on the DGX versus included in the ZIP. Hashes identify artifacts, not proof of remote execution or registration time.

---

## 11. Implementation milestones and acceptance tests

### Milestone 1: source audit, adapter, and runnable smoke test

Inspect Phase One and upstream code. Write compatibility/attribution notes, the dependency boundary, model-revision lock, synthetic generator, original-activation/no-op patch path, and AV/AR adapters. Use tiny randomly initialized fixtures for CPU tests; do not label their outputs Qwen results.

**Minimum tests:** hook indexing and single-site mutation; all other token vectors unchanged; native-dtype raw restoration; hook cleanup after exceptions; no stale state between samples; padding/position handling; metadata mismatch rejection; required AR-head loading; zero/nonfinite norm rejection; AV/AR isolation from prompt/truth; answer-tokenization correctness; safe artifact loading; overwrite refusal. Add a regression proving two embedding injections do not reuse an incorrect cached AV result.

Deliver the isolated environment instructions and one smoke command that accepts revision locks/model paths. It should report package compatibility and measured evidence, or fail with a precise actionable diagnostic. Download credentials or hardware access that are unavailable must remain explicit blockers.

### Milestone 2: one pilot

Implement all planned preservation controls, text-edit coverage accounting, resource measurement, grouped data splits, metrics and report audit. Run or hand off the 128-group pilot. Publish a pilot decision: keep/stop, intervention sensitivity, edit eligibility, final operating choices, projected validation cost, and any deviations from this brief.

Do not add a new architecture, train a new NLA, or shop for a favorable checkpoint after a negative pilot. A source/API bug may be repaired with a regression test and a visibly versioned rerun of the affected pilot only.

### Milestone 3: one locked validation, then write-up

Freeze the primary, all operational settings, coverage rules, metrics and thresholds before evaluating the 512 held-out groups. Do not generate/read the validation outcomes during pilot tuning. Execute or hand off one validation command, audit its evidence, and produce a demonstration and technical report even if the result is negative.

No automatic continuation to Ouro, Huginn, extra layers, larger models, different geometries, or new training. Those are separate research decisions after this phase closes.

### Required final Codex handoff

Provide changed files and actual git status; exact commands tested; passed/failed/skipped tests with reasons; a source/compatibility lock; the next single copy-pasteable DGX command; expected report filenames; and the guarantees that remain untested. Use an existing supplied workspace/branch when required by the environment, and do not claim a remote commit/merge unless a successful remote read verifies it. A pull request can be prepared when authorized; auto-merging is not part of this brief.

---

## 12. Claims the eventual report may and may not make

| Observation | Defensible interpretation | Prohibited upgrade |
|---|---|---|
| High cosine reconstruction | Similar activation direction. | The model's reasoning was recovered. |
| P2 preserves measured answers | Selected-vector reconstruction preserved behavior on this assay. | The whole hidden state or KV cache was compressed. |
| Correct edits outperform wrong-variable controls | Evidence for targeted influence under the specified intervention. | A unique causal variable or all neuron meanings were discovered. |
| Minimal edits are often unavailable | Limited explicit-variable coverage at this task/site. | Ground-truth text inserted manually was read from the activation. |
| Numerical baseline is as good or better | Generic reconstruction is competitive under the stated budget. | Language is useless for all interpretability. |
| Single-site patch is ineffective | This site/assay did not reveal a measurable contribution. | The model has no relevant internal representation. |
| Small stored description | Small selected-vector payload. | Low total cost despite AV/AR weights, norm, context, calls and metadata. |

The different-spaces ambition remains relevant, but comparing descriptions of a representation is not yet changing the target model's reasoning architecture. Establish this first transfer honestly before claiming a general mathematical lingua franca.

---

## 13. Primary sources and how to use them

Sources checked while drafting; resolve immutable upstream revisions before execution. The design choices in Sections 3–12 are **proposals**, not results reported by these sources.

- **[S1] Project requirements and Phase-One closeout.** `seanm27lol/universa-recurrent`, inspected HEAD `ecc8d79873b6f76b4e896986092c7ce8909cdf75`:
  - https://github.com/seanm27lol/universa-recurrent/blob/ecc8d79873b6f76b4e896986092c7ce8909cdf75/AGENTS.md
  - https://github.com/seanm27lol/universa-recurrent/blob/ecc8d79873b6f76b4e896986092c7ce8909cdf75/docs/phase_one_technical_report.md
  - https://github.com/seanm27lol/universa-recurrent/blob/ecc8d79873b6f76b4e896986092c7ce8909cdf75/docs/phase_one_results.md
- **[S2] Qwen team model card.** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **[S3] Released AV model card and file inventory.** https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av
- **[S4] Released AR model card and file inventory.** https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar
- **[S5] Author-maintained lightweight inference recipe.** https://github.com/kitft/nla-inference
- **[S6] Inference architecture, normalization and reconstruction details.** https://github.com/kitft/nla-inference/blob/main/README.md
- **[S7] Author-maintained training repository and inference documentation.** https://github.com/kitft/natural_language_autoencoders and https://github.com/kitft/natural_language_autoencoders/blob/main/docs/inference.md
- **[S8] Fraser-Taliente, Kantamneni, Ong et al. (2026), _Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations_.** https://transformer-circuits.pub/2026/nla/ . Use its reconstruction, intervention precedent and limitations; do not import its reported performance to this task.
- **[S9] Zhang and Nanda, _Towards Best Practices of Activation Patching in Language Models: Metrics and Methods_.** https://arxiv.org/abs/2309.16042 . Ground intervention and metric choices; do not treat patching as automatic proof of a complete causal mechanism.
- **[S10] NVIDIA, DGX Spark hardware overview.** https://docs.nvidia.com/dgx/dgx-spark/hardware.html . Hardware capacity is not evidence that this exact software stack has run.
- **[S11] OpenAI, project instructions with AGENTS.md.** https://developers.openai.com/codex/guides/agents-md . Read applicable repository instructions alongside this explicitly supplied brief; do not replace the project's enduring rules.

**End condition:** a reproducible, bounded answer about one released NLA pair on one open-weight model. Success, failure, or insufficient intervention coverage must all lead to a documented result, not an unbounded sequence of new benchmarks.
