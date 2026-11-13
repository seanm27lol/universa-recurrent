"""Fixed endpoint adapter for existing neural generation, CE, readout and replay.

Generation sees only exported prompts. Reference execution and changed-register
labels enter the independent scorer after predictions have been saved.
"""
from pathlib import Path

from . import data as D, evaluate as E, reconstruct
from .answer_analysis import AnswerMaskedDataset
from .diagnose_program_results import teacher_forced_answer_cross_entropy
from .evaluate_state_path import evaluate_state_path
from .independent.score_initial_state_study import score
from .initial_state_study import POPULATIONS


def evaluate(checkpoint, data, tokenizer, output):
    from .run_initial_state_study import write_json
    import json

    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path, digest = checkpoint["path"], checkpoint["sha256"]
    model, _ = E.load_native_model(path, expected_sha256=digest, device="cpu")
    tokens = D.load_gpt2_tokenizer(tokenizer)
    report = {"format": "native-initial-state-evaluation-v2", "checkpoint_sha256": digest,
              "complete": False, "populations": {},
              "scope": "CPU FP32 endpoint evaluation; no model intervention or shaping"}
    for population in POPULATIONS:
        prompts = [json.loads(line) for line in (data / "prompts" / f"{population}.jsonl").read_text().splitlines()]
        if len(prompts) != 128 or any(set(p) != {"example_id", "prompt"} for p in prompts):
            raise ValueError("generation requires exactly 128 prompt-only records")
        predictions = []
        for prompt in prompts:
            generated = E.generate_prompt(model, tokens, prompt["prompt"], max_new_tokens=48,
                                           eos_token_id=D.EOS_TOKEN_ID, use_cache=True, prompt_policy="error")
            predictions.append({"example_id": prompt["example_id"], **generated})
        saved = output / f"{population}.predictions.jsonl"
        D._atomic_jsonl(saved, predictions)
        scoring = score(data / "evaluation" / f"{population}.jsonl", saved, output / f"{population}.score.json")
        arrays = data / "dev" if population == "dev" else data / "evaluation_arrays" / population
        dataset = AnswerMaskedDataset(arrays, mask_directory=data / "answer_masks" / population)
        ce = teacher_forced_answer_cross_entropy(model, dataset, batch_size=8, max_batches=None)
        state = evaluate_state_path(path, arrays, output / f"{population}.state.json",
                                    expected_sha256=digest, batch_size=8, max_batches=None, position_policy="document")
        report["populations"][population] = {"score": scoring, "answer_ce": ce, "state_readout": state}
        if not ce["complete"] or not state["complete"]:
            raise RuntimeError("incomplete endpoint diagnostics")
    replay = reconstruct.load_replay_spec(data / "replay.json")
    report["reconstruction"] = reconstruct.reconstruct_checkpoint(path, replay, expected_sha256=digest, device="cpu")
    # Strict-allclose outcomes remain visible even when scale-based replay agrees.
    report["complete"] = len(report["populations"]) == len(POPULATIONS) and report["reconstruction"]["agreement"]
    if D.sha256_file(path) != digest:
        raise RuntimeError("checkpoint changed during evaluation")
    write_json(output / "evaluation.json", report)
    return report
