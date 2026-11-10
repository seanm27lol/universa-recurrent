"""Compare standalone zero-based and packed-position LM losses and gradients on CPU."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import MODEL_INPUT_ARRAYS, PackedDataset, load_gpt2_tokenizer, sha256_file
from .evaluate import load_native_model


POSITION_NAMES = ("zero_position", "actual_packed_position")
GROUP_NAMES = (
    "position_embedding",
    "token_embedding",
    "attention/core",
    "experts/router",
    "state_interfaces",
    "output_head",
)
_CORE_MODULES = {"attention", "attentions", "transform", "transforms"}
_EXPERT_MODULES = {"experts", "router"}
_STATE_MODULES = {
    "h_reader", "h_writer", "slot_initializers", "slot_reader", "slot_writer",
    "slot_readers", "slot_writers", "slot_codes", "state_probes",
}


@dataclass(frozen=True)
class Example:
    example_id: str
    doc_id: int
    row: int
    start: int
    end: int
    actual_starting_position: int
    prompt_tokens: int
    answer_tokens: int
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class UnitExample:
    example: Example
    start: int
    end: int


def parameter_group(name: str) -> str:
    root = name.split(".", 1)[0]
    if root in ("position_embedding", "token_embedding", "output_head", "output_norm"):
        return "output_head" if root == "output_norm" else root
    if root in _CORE_MODULES:
        return "attention/core"
    if root in _EXPERT_MODULES:
        return "experts/router"
    if root in _STATE_MODULES:
        return "state_interfaces"
    raise ValueError(f"unassigned parameter {name!r}")


def parameter_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups = {name: [] for name in GROUP_NAMES}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[parameter_group(name)].append((name, parameter))
    return groups


def shifted_masks(
    labels: torch.Tensor,
    answer_targets: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.shape != answer_targets.shape or labels.shape != key_padding_mask.shape:
        raise ValueError("labels, answer targets, and padding mask must have identical [B,T] shapes")
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("causal shifting requires [B,T] labels with T >= 2")
    full = labels[:, 1:].ne(-100) & ~key_padding_mask[:, 1:].bool()
    answer = answer_targets[:, 1:].bool() & full
    return full, answer


def masked_ce_sum(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("aligned logits, targets, and mask must have matching non-class dimensions")
    selected_targets = targets[mask]
    selected_logits = logits[mask]
    if selected_targets.numel() == 0 or selected_targets.lt(0).any():
        raise ValueError("a supervised example must have valid, non-ignored targets")
    return F.cross_entropy(selected_logits.float(), selected_targets.long(), reduction="sum")


def _loss_record(loss: torch.Tensor, count: int) -> dict:
    value = float(loss.detach())
    return {"sum": value, "count": count, "mean": value / count}


class GradientAccumulator:
    def __init__(self, model: torch.nn.Module):
        self.groups = parameter_groups(model)
        self.parameters = [parameter for values in self.groups.values() for _, parameter in values]
        self.indices = {}
        offset = 0
        for group, values in self.groups.items():
            self.indices[group] = list(range(offset, offset + len(values)))
            offset += len(values)
        self.answer_gradients = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.full_gradients = [torch.zeros_like(parameter) for parameter in self.parameters]

    def update(self, answer_loss: torch.Tensor, full_loss: torch.Tensor) -> None:
        answer_gradients = torch.autograd.grad(
            answer_loss, self.parameters, retain_graph=True, allow_unused=True,
        )
        full_gradients = torch.autograd.grad(
            full_loss, self.parameters, retain_graph=False, allow_unused=True,
        )
        for group in GROUP_NAMES:
            for index in self.indices[group]:
                answer = answer_gradients[index]
                full = full_gradients[index]
                if answer is not None:
                    if not torch.isfinite(answer).all():
                        raise FloatingPointError(f"nonfinite answer gradient in {group}")
                    self.answer_gradients[index].add_(answer.detach())
                if full is not None:
                    if not torch.isfinite(full).all():
                        raise FloatingPointError(f"nonfinite full-LM gradient in {group}")
                    self.full_gradients[index].add_(full.detach())

    def summary(self) -> dict:
        result = {}
        sums = {group: {"answer_square": 0.0, "full_square": 0.0, "dot": 0.0}
                for group in (*GROUP_NAMES, "all_parameters")}
        for group in GROUP_NAMES:
            for index in self.indices[group]:
                answer = self.answer_gradients[index]
                full = self.full_gradients[index]
                sums[group]["answer_square"] += float(torch.sum(answer * answer, dtype=torch.float64))
                sums[group]["full_square"] += float(torch.sum(full * full, dtype=torch.float64))
                sums[group]["dot"] += float(torch.sum(answer * full, dtype=torch.float64))
        for name in ("answer_square", "full_square", "dot"):
            sums["all_parameters"][name] = sum(sums[group][name] for group in GROUP_NAMES)
        for group in (*GROUP_NAMES, "all_parameters"):
            values = sums[group]
            answer_norm = math.sqrt(values["answer_square"])
            full_norm = math.sqrt(values["full_square"])
            cosine = None
            if answer_norm > 0.0 and full_norm > 0.0:
                cosine = max(-1.0, min(1.0, values["dot"] / (answer_norm * full_norm)))
            result[group] = {
                "answer_only_gradient_l2": answer_norm,
                "full_lm_gradient_l2": full_norm,
                "answer_only_full_lm_cosine": cosine,
            }
            if group != "all_parameters":
                result[group]["parameter_count"] = sum(
                    parameter.numel() for _, parameter in self.groups[group]
                )
        return result


def load_examples(data_root: str | Path, tokenizer) -> tuple[PackedDataset, list[Example]]:
    root = Path(data_root)
    dataset = PackedDataset(root, "train")
    records = {
        record["example_id"]: record
        for record in map(json.loads, (root / "structured" / "train.jsonl").read_text().splitlines())
    }
    consumed = [json.loads(line) for line in (root / "train_consumed_documents.jsonl").read_text().splitlines()]
    examples = []
    for item in consumed:
        doc_id = int(item["doc_id"])
        record = records[item["example_id"]]
        prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False).ids
        answer_ids = tokenizer.encode(record["target_text"], add_special_tokens=False).ids + [50256]
        expected_ids = tuple(prompt_ids + answer_ids)
        locations = []
        for row in range(len(dataset)):
            columns = torch.from_numpy(
                ((dataset.arrays["doc_ids"][row] == doc_id) & ~dataset.arrays["key_padding_mask"][row]).copy()
            ).nonzero().flatten().tolist()
            if columns:
                locations.append((row, columns))
        if len(locations) != 1:
            raise ValueError(f"{item['example_id']}: expected one intact packed row, found {len(locations)}")
        row, columns = locations[0]
        if columns != list(range(columns[0], columns[-1] + 1)):
            raise ValueError(f"{item['example_id']}: packed document is not contiguous")
        start, end = columns[0], columns[-1] + 1
        actual_ids = tuple(map(int, dataset.arrays["input_ids"][row, start:end]))
        if actual_ids != expected_ids or len(actual_ids) != int(item["tokens"]):
            raise ValueError(f"{item['example_id']}: structured and packed tokenization differ")
        positions = dataset.arrays["positions"][row, start:end]
        if positions.tolist() != list(range(start, end)):
            raise ValueError(f"{item['example_id']}: packed positions do not continue across the row")
        expected_labels = [-100, *expected_ids[1:]]
        if dataset.arrays["labels"][row, start:end].tolist() != expected_labels:
            raise ValueError(f"{item['example_id']}: packed causal labels differ from its document")
        examples.append(Example(
            example_id=item["example_id"], doc_id=doc_id, row=row, start=start, end=end,
            actual_starting_position=int(positions[0]), prompt_tokens=len(prompt_ids),
            answer_tokens=len(answer_ids), token_ids=expected_ids,
        ))
    examples.sort(key=lambda example: example.doc_id)
    if len(examples) != 32 or {example.doc_id for example in examples} != set(range(32)):
        raise ValueError("the acquisition diagnostic requires exactly doc_ids 0 through 31")
    return dataset, examples


def _answer_targets(length: int, unit_examples: Sequence[UnitExample]) -> torch.Tensor:
    result = torch.zeros((1, length), dtype=torch.bool)
    for item in unit_examples:
        answer_start = item.start + item.example.prompt_tokens
        result[0, answer_start:item.end] = True
    return result


def _run_unit(
    model: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    unit_examples: Sequence[UnitExample],
    full_token_total: int,
    answer_token_total: int,
    gradients: GradientAccumulator,
) -> dict[str, dict]:
    output = model(**inputs)
    shifted_logits = output.logits[:, :-1]
    shifted_labels = labels[:, 1:]
    answer_targets = _answer_targets(labels.shape[1], unit_examples)
    full_mask, answer_mask = shifted_masks(labels, answer_targets, inputs["key_padding_mask"])
    losses = {}
    full_losses = []
    answer_losses = []
    for item in unit_examples:
        example = item.example
        local = slice(item.start, item.end - 1)
        local_logits = shifted_logits[:, local]
        local_labels = shifted_labels[:, local]
        local_full_mask = full_mask[:, local]
        local_answer_mask = answer_mask[:, local]
        full_count = int(local_full_mask.sum())
        answer_count = int(local_answer_mask.sum())
        if full_count != example.token_count - 1 or answer_count != example.answer_tokens:
            raise AssertionError(f"{example.example_id}: causal mask counts are misaligned")
        full_loss = masked_ce_sum(local_logits, local_labels, local_full_mask)
        answer_loss = masked_ce_sum(local_logits, local_labels, local_answer_mask)
        full_losses.append(full_loss)
        answer_losses.append(answer_loss)
        losses[example.example_id] = {
            "full_lm": _loss_record(full_loss, full_count),
            "answer": _loss_record(answer_loss, answer_count),
        }
    gradients.update(
        sum(answer_losses) / answer_token_total,
        sum(full_losses) / full_token_total,
    )
    return losses


def _aggregate(per_example: Mapping[str, dict], examples: Sequence[Example]) -> dict:
    result = {}
    for objective in ("full_lm", "answer"):
        count = sum(per_example[example.example_id][objective]["count"] for example in examples)
        loss_sum = sum(per_example[example.example_id][objective]["sum"] for example in examples)
        result[objective] = {"sum": loss_sum, "count": count, "mean": loss_sum / count}
    return result


def _position_pass(model, units, examples, full_token_total, answer_token_total) -> dict:
    gradients = GradientAccumulator(model)
    per_example = {}
    for inputs, labels, unit_examples in units:
        losses = _run_unit(
            model, inputs, labels, unit_examples, full_token_total, answer_token_total, gradients,
        )
        if per_example.keys() & losses.keys():
            raise AssertionError("an example was scored more than once in one position pass")
        per_example.update(losses)
    if set(per_example) != {example.example_id for example in examples}:
        raise AssertionError("position pass did not score every train example exactly once")
    return {"losses": _aggregate(per_example, examples), "gradient_by_parameter_group": gradients.summary()}, per_example


def _zero_units(examples: Sequence[Example]):
    for example in examples:
        ids = torch.tensor([example.token_ids], dtype=torch.long)
        labels = ids.clone()
        labels[0, 0] = -100
        inputs = {
            "input_ids": ids,
            "positions": torch.arange(example.token_count, dtype=torch.long).unsqueeze(0),
            "doc_ids": torch.zeros_like(ids),
            "key_padding_mask": torch.zeros_like(ids, dtype=torch.bool),
        }
        yield inputs, labels, [UnitExample(example, 0, example.token_count)]


def _actual_units(dataset: PackedDataset, examples: Sequence[Example]):
    for row in range(len(dataset)):
        batch = dataset.batch([row])
        unit_examples = [
            UnitExample(example, example.start, example.end)
            for example in examples if example.row == row
        ]
        yield {name: batch[name] for name in MODEL_INPUT_ARRAYS}, batch["labels"], unit_examples


def diagnose_checkpoint(checkpoint, expected_sha256, dataset, examples, saved_answer_ce) -> dict:
    model, descriptor = load_native_model(checkpoint, expected_sha256=expected_sha256, device="cpu")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("the diagnostic requires CPU-only parameters")
    full_token_total = sum(example.token_count - 1 for example in examples)
    answer_token_total = sum(example.answer_tokens for example in examples)
    zero, zero_examples = _position_pass(
        model, _zero_units(examples), examples, full_token_total, answer_token_total,
    )
    actual, actual_examples = _position_pass(
        model, _actual_units(dataset, examples), examples, full_token_total, answer_token_total,
    )
    computed = zero["losses"]["answer"]
    tolerance = {"absolute": 1e-8, "relative": 1e-10}
    matches = (
        computed["count"] == saved_answer_ce["count"]
        and math.isclose(computed["sum"], saved_answer_ce["sum"],
                         rel_tol=tolerance["relative"], abs_tol=tolerance["absolute"])
        and math.isclose(computed["mean"], saved_answer_ce["mean"],
                         rel_tol=tolerance["relative"], abs_tol=tolerance["absolute"])
    )
    example_results = []
    for example in examples:
        zero_loss = zero_examples[example.example_id]
        actual_loss = actual_examples[example.example_id]
        example_results.append({
            "example_id": example.example_id,
            "doc_id": example.doc_id,
            "packed_row": example.row,
            "packed_column_start": example.start,
            "packed_column_end_exclusive": example.end,
            "actual_starting_position": example.actual_starting_position,
            "row_first": example.actual_starting_position == 0,
            "zero_position": zero_loss,
            "actual_packed_position": actual_loss,
            "actual_minus_zero": {
                "full_lm_mean": actual_loss["full_lm"]["mean"] - zero_loss["full_lm"]["mean"],
                "answer_mean": actual_loss["answer"]["mean"] - zero_loss["answer"]["mean"],
            },
        })
    restored = model.state_dict()
    parameters_unchanged = all(
        torch.equal(restored[name].detach().cpu(), expected.detach().cpu())
        for name, expected in descriptor["model"].items()
    ) and all(parameter.grad is None for parameter in model.parameters())
    if not parameters_unchanged:
        raise AssertionError("model parameters or .grad fields changed during the diagnostic")
    post_sha256 = sha256_file(checkpoint)
    if post_sha256 != expected_sha256:
        raise AssertionError("checkpoint bytes changed during the diagnostic")
    return {
        "checkpoint": {
            "path": str(checkpoint), "sha256": expected_sha256, "post_diagnostic_sha256": post_sha256,
            "step": descriptor["step"], "mode": model.config.mode,
        },
        "device": "cpu",
        "parameter_updates": 0,
        "parameters_unchanged": parameters_unchanged,
        "position_results": {"zero_position": zero, "actual_packed_position": actual},
        "saved_answer_ce_reproduction": {
            "saved": saved_answer_ce,
            "computed_zero_position": computed,
            "sum_difference_computed_minus_saved": computed["sum"] - saved_answer_ce["sum"],
            "mean_difference_computed_minus_saved": computed["mean"] - saved_answer_ce["mean"],
            "tolerance": tolerance,
            "matches": matches,
        },
        "examples": example_results,
    }


def diagnose(run_directory, data_directory, tokenizer_path=None) -> dict:
    run = Path(run_directory)
    data = Path(data_directory)
    protocol = json.loads((run / "protocol.json").read_text())
    provenance = json.loads((data / "source_provenance.json").read_text())
    tokenizer_path = tokenizer_path or provenance["tokenizer_path"]
    tokenizer = load_gpt2_tokenizer(tokenizer_path)
    dataset, examples = load_examples(data, tokenizer)
    arms = []
    for arm in protocol["arms"]:
        status = json.loads((run / "arms" / arm / "status.json").read_text())
        checkpoint = status["checkpoint"]
        state_path = run / "arms" / arm / "train_state_path.json"
        saved_answer = json.loads(state_path.read_text())["answer_token_cross_entropy"]["train"]
        result = diagnose_checkpoint(
            checkpoint["path"], checkpoint["sha256"], dataset, examples, saved_answer,
        )
        if result["checkpoint"]["mode"] != arm or result["checkpoint"]["step"] != protocol["max_updates"]:
            raise ValueError(f"{arm}: checkpoint does not match the acquisition endpoint")
        result["saved_answer_ce_reproduction"]["source"] = str(state_path)
        arms.append(result)
        print(json.dumps({
            "arm": arm,
            "zero_answer_ce": result["position_results"]["zero_position"]["losses"]["answer"]["mean"],
            "actual_answer_ce": result["position_results"]["actual_packed_position"]["losses"]["answer"]["mean"],
            "saved_answer_ce_matches": result["saved_answer_ce_reproduction"]["matches"],
        }, sort_keys=True), flush=True)
        gc.collect()
    return {
        "format": "native-loss-gradient-diagnostic-v2",
        "run": str(run),
        "data": str(data),
        "tokenizer": str(tokenizer_path),
        "device": "cpu",
        "example_count": len(examples),
        "packed_rows": len(dataset),
        "row_first_examples": sum(example.actual_starting_position == 0 for example in examples),
        "full_lm_token_count": sum(example.token_count - 1 for example in examples),
        "answer_token_count": sum(example.answer_tokens for example in examples),
        "objective_definitions": {
            "zero_position": "standalone prompt+answer+EOS with positions 0..T-1 and no padding",
            "actual_packed_position": "original intact packed row with saved positions, document-blocked attention, and shifted label masks",
            "full_lm": "sum CE over every within-document next-token target, including answer and EOS; first document token and padding excluded",
            "answer": "sum CE over answer tokens and EOS only, aligned to preceding-position logits",
            "aggregate_gradients": "separate gradients of global token-count-mean answer-only and full-LM CE; auxiliary route/state/balance losses excluded",
            "cosine": "cosine between aggregate answer-only and aggregate full-LM gradients within each position regime and parameter group",
            "unused_parameters": "unused autograd results contribute zero norm; cosine is null when either group norm is zero",
        },
        "arms": arms,
        "evidence_boundary": "diagnostic measurements on the completed 32-example fit only; no parameter update, architecture change, or generalization claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.run) / "loss_gradient_diagnostic_v2.json"
    result = diagnose(args.run, args.data, args.tokenizer)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "arms": len(result["arms"])}, sort_keys=True))


if __name__ == "__main__":
    main()
