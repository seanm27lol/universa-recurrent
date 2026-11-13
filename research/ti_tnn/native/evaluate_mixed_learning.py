"""Language/code capability only; synthetic scores and causal claims stay separate."""
from __future__ import annotations

import json
from pathlib import Path

from . import data as D
from .mixed_learning import CAPABILITY_DOCUMENTS, GENERATION_DOCUMENTS, CONTINUATION_TOKENS, PROMPT_TOKENS


def score_continuations(references, predictions):
    """Score saved token continuations with missing/short outputs as failures.

    Exact means all 32 reference tokens, not successful program execution. Every
    reference retains 32 token decisions. Duplicate/unknown ids are integrity errors.
    """
    expected = {r["example_id"]: r for r in references}
    if len(expected) != len(references) or not expected:
        raise ValueError("references must have unique nonempty identities")
    supplied = {}
    for row in predictions:
        key = row["example_id"]
        if key not in expected or key in supplied:
            raise ValueError("unknown or duplicate prediction identity")
        supplied[key] = row
    exact = token_correct = missing = malformed = short = eos = 0
    for key, reference in expected.items():
        target = reference["target_ids"]
        if (len(target) != CONTINUATION_TOKENS
                or any(type(v) is not int or not 0 <= v < D.VOCAB_SIZE for v in target)):
            raise ValueError("reference endpoint changed")
        row = supplied.get(key)
        if row is None:
            missing += 1
            continue
        ids = row.get("generated_ids")
        if (not isinstance(ids, list) or len(ids) > CONTINUATION_TOKENS
                or any(type(v) is not int or not 0 <= v < D.VOCAB_SIZE for v in ids)):
            malformed += 1
            continue
        exact += ids == target
        short += len(ids) < CONTINUATION_TOKENS
        eos += D.EOS_TOKEN_ID in ids
        token_correct += sum(a == b for a, b in zip(ids, target))
    return {"examples": len(expected), "exact": exact, "exact_rate": exact / len(expected),
        "token_correct": token_correct, "token_decisions": len(expected) * CONTINUATION_TOKENS,
        "missing": missing, "malformed": malformed, "short_outputs": short,
        "outputs_containing_eos": eos, "complete": missing == 0,
        "claim": "fixed-token prefix continuation; not semantic or executable-code correctness"}


def generate_continuations(model, prompts):
    """Only prompt tokens enter neural inference; a common 32-token bound for all."""
    import torch
    from .generation import greedy_generate

    predictions = []
    seen = set()
    for row in prompts:
        if set(row) != {"example_id", "input_ids"} or len(row["input_ids"]) != PROMPT_TOKENS:
            raise ValueError("expected prompt-only 128-token record")
        if row["example_id"] in seen:
            raise ValueError("duplicate prompt")
        seen.add(row["example_id"])
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=next(model.parameters()).device)
        output = greedy_generate(model, ids, max_new_tokens=CONTINUATION_TOKENS, eos_token_id=D.EOS_TOKEN_ID,
                                 pad_token_id=D.EOS_TOKEN_ID, use_cache=True)
        predictions.append({"example_id": row["example_id"], "generated_ids": output[0, PROMPT_TOKENS:].tolist()})
    return predictions


def evaluate_capability(model, preparation, output):
    """Called only in a separately authorized endpoint evaluation; never by prepare."""
    from .evaluate import teacher_forced_evaluate

    preparation, output = Path(preparation), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((preparation / "manifest.json").read_text())
    for relative, digest in manifest["artifacts"].items():
        if relative.startswith("capability/"):
            D.verify_file(preparation / relative, digest)
    report = {}
    for source in ("wikipedia", "code"):
        folder = preparation / "capability" / source
        references = [json.loads(x) for x in (folder / "continuations.jsonl").read_text().splitlines()]
        prompts = [json.loads(x) for x in (folder / "prompts.jsonl").read_text().splitlines()]
        if len(references) != GENERATION_DOCUMENTS or prompts != [
            {"example_id": r["example_id"], "input_ids": r["prompt_ids"]} for r in references
        ]:
            raise ValueError("capability population/prompt binding changed")
        dataset = D.PackedDataset(folder / "teacher_forced")
        if len(dataset) != CAPABILITY_DOCUMENTS:
            raise ValueError("capability teacher-forced denominator changed")
        predictions = generate_continuations(model, prompts)
        D._atomic_jsonl(output / f"{source}.predictions.jsonl", predictions)
        report[source] = {"continuation": score_continuations(references, predictions),
            "teacher_forced": teacher_forced_evaluate(model, dataset,
                                                       batch_size=8, max_batches=None)}
    D._atomic_json(output / "capability.json", report)
    return report


def evaluate_endpoint(checkpoint, preparation, tokenizer, output):
    """Compose existing CPU endpoint diagnostics without starting a reservation.

    The authorized caller must validate training/source/sampler/config identities
    and the cumulative allocation using the existing runner before reaching here.
    This function binds the prepared bytes and checkpoint, not authorization.
    """
    import time
    from .evaluate import load_native_model
    from .evaluate_initial_state_study import evaluate as evaluate_registers

    started = time.perf_counter()
    preparation, output = Path(preparation), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((preparation / "manifest.json").read_text())
    for relative, digest in manifest["artifacts"].items():
        D.verify_file(preparation / relative, digest)
    D.verify_file(tokenizer, D.TOKENIZER_SPEC["sha256"])
    model, _ = load_native_model(checkpoint["path"], expected_sha256=checkpoint["sha256"], device="cpu")
    capability = evaluate_capability(model, preparation, output / "capability")
    del model
    registers = evaluate_registers(checkpoint, preparation / "register_monitor", tokenizer, output / "register_monitor")
    report = {"format": "native-mixed-learning-evaluation-v1", "checkpoint_sha256": checkpoint["sha256"],
        "preparation_sha256": D.sha256_file(preparation / "manifest.json"),
        "capability": capability, "synthetic_diagnostics": registers,
        "causal_interpretation": "not run; readout quality and answer correctness are separate claims",
        "register_population_role": "historical fixed monitoring; train_anchor/train_sample refer to #80, not this run",
        "cpu_side_wall_seconds": time.perf_counter() - started,
        "complete": registers["complete"] and all(v["continuation"]["complete"]
            and v["teacher_forced"]["complete"] for v in capability.values())}
    D.verify_file(checkpoint["path"], checkpoint["sha256"])
    D._atomic_json(output / "evaluation.json", report)
    return report
