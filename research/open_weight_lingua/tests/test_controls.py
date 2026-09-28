import json

import pytest
import torch
from safetensors.torch import save as serialize_tensors

from open_weight_lingua.artifacts import sha256_file
from open_weight_lingua.controls import (
    BYTES_PER_COEFFICIENT,
    FIT_SIDECAR_NAME,
    FIT_TENSORS_NAME,
    fit_pca,
    load_fit,
    reconstruct,
    save_fit,
)
from open_weight_lingua.geometry import direction_metrics, unit_direction

WIDTH = 64
PLANTED_RANK = 4
SUBSPACE_SEED = 7


def make_directions(n=128, noise=0.0, seed=1234):
    """Synthetic unlabeled vectors near one planted low-rank subspace."""
    subspace_generator = torch.Generator().manual_seed(SUBSPACE_SEED)
    subspace = torch.linalg.qr(
        torch.randn(WIDTH, PLANTED_RANK, generator=subspace_generator)
    ).Q
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(n, PLANTED_RANK, generator=generator) @ subspace.T
    if noise:
        vectors = vectors + noise * torch.randn(n, WIDTH, generator=generator)
    return vectors


def test_recovery_is_near_identity_when_budget_covers_planted_rank():
    fit = fit_pca(make_directions())
    assert fit.fitted_rank == PLANTED_RANK
    assert fit.width == WIDTH and fit.dtype == torch.float32
    for vector in make_directions(n=8, seed=999):
        direction, accounting = reconstruct(fit, vector, text_bytes=10_000)
        assert accounting["rank_used"] == PLANTED_RANK
        assert direction_metrics(vector, direction)["cosine"] > 0.999


def test_noisy_recovery_beats_chance_under_a_tight_budget():
    fit = fit_pca(make_directions(noise=0.05))
    assert fit.fitted_rank >= PLANTED_RANK
    vector = make_directions(n=1, noise=0.05, seed=555)[0]
    direction, accounting = reconstruct(
        fit, vector, text_bytes=BYTES_PER_COEFFICIENT * PLANTED_RANK
    )
    assert accounting["rank_used"] == PLANTED_RANK
    scores = direction_metrics(vector, direction)
    chance = direction_metrics(
        vector, torch.randn(WIDTH, generator=torch.Generator().manual_seed(77))
    )
    assert scores["cosine"] > 0.9
    assert scores["cosine"] > chance["cosine"] + 0.5
    assert scores["unit_direction_squared_l2"] == pytest.approx(
        2 * (1 - scores["cosine"]), rel=1e-5
    )


def test_byte_budget_rule_caps_rank_and_accounting_sums():
    fit = fit_pca(make_directions())
    vector = make_directions(n=1, seed=31337)[0]
    cases = [
        (10_000, fit.fitted_rank),  # fitted rank caps below the language budget
        (2 * fit.fitted_rank, fit.fitted_rank),
        (2 * fit.fitted_rank - 1, fit.fitted_rank - 1),
        (3, 1),  # text bytes smaller than 2 * fitted_rank
        (1, 0),
        (0, 0),
    ]
    for text_bytes, expected_rank in cases:
        _, accounting = reconstruct(fit, vector, text_bytes=text_bytes)
        assert accounting["rank_used"] == expected_rank
        assert accounting["text_budget_bytes"] == text_bytes
        assert accounting["fitted_rank"] == fit.fitted_rank
        assert (
            accounting["coefficient_bytes"] == BYTES_PER_COEFFICIENT * expected_rank
        )
        assert accounting["coefficient_bytes"] <= text_bytes
        assert (
            accounting["unused_budget_bytes"]
            == text_bytes - accounting["coefficient_bytes"]
        )
        assert accounting["budget_interpretation"] == "no-more-than-budget"
        assert accounting["baseline_fit_identity"] == fit.identity


def test_coefficients_are_quantized_to_float16():
    fit = fit_pca(make_directions())
    vector = make_directions(n=1, seed=222)[0]
    direction, accounting = reconstruct(fit, vector, text_bytes=10_000)
    rank = accounting["rank_used"]
    raw = fit.basis[:rank] @ (unit_direction(vector) - fit.mean)
    quantized = raw.to(torch.float16)
    assert quantized.dtype == torch.float16
    assert not torch.equal(quantized.float(), raw)
    expected = unit_direction(fit.mean + fit.basis[:rank].T @ quantized.float())
    assert torch.equal(direction, expected)


def test_fit_is_deterministic_across_calls_and_input_forms():
    vectors = make_directions()
    first, second = fit_pca(vectors), fit_pca(list(vectors))
    assert first.identity == second.identity
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.basis, second.basis)
    assert (first.fitted_rank, first.width, first.dtype) == (
        second.fitted_rank,
        second.width,
        second.dtype,
    )


def test_shared_storage_accounting_declares_dtype_and_bytes():
    fit = fit_pca(make_directions())
    _, accounting = reconstruct(fit, torch.ones(WIDTH), text_bytes=4)
    assert accounting["shared_dtype"] == "float32"
    assert accounting["shared_mean_basis_bytes"] == WIDTH * (1 + fit.fitted_rank) * 4
    assert accounting["coefficient_dtype"] == "float16"
    assert accounting["bytes_per_coefficient"] == BYTES_PER_COEFFICIENT


def test_save_load_round_trip_preserves_identity(tmp_path):
    fit = fit_pca(make_directions())
    directory = tmp_path / "fit"
    sidecar = save_fit(fit, directory)
    assert sidecar["baseline_fit_identity"] == fit.identity
    on_disk = json.loads((directory / FIT_SIDECAR_NAME).read_text())
    assert on_disk["baseline_fit_identity"] == fit.identity
    loaded = load_fit(directory)
    assert loaded.identity == fit.identity
    assert torch.equal(loaded.mean, fit.mean)
    assert torch.equal(loaded.basis, fit.basis)
    assert (loaded.fitted_rank, loaded.width, loaded.dtype) == (
        fit.fitted_rank,
        fit.width,
        fit.dtype,
    )
    vector = make_directions(n=1, seed=4242)[0]
    assert torch.equal(
        reconstruct(loaded, vector, text_bytes=64)[0],
        reconstruct(fit, vector, text_bytes=64)[0],
    )
    with pytest.raises(FileExistsError):
        save_fit(fit, directory)


def _copy_fit(tmp_path, name, source, *, sidecar_updates=None, tensors=None):
    directory = tmp_path / name
    directory.mkdir()
    payload = (
        serialize_tensors(tensors)
        if tensors is not None
        else (source / FIT_TENSORS_NAME).read_bytes()
    )
    (directory / FIT_TENSORS_NAME).write_bytes(payload)
    sidecar = json.loads((source / FIT_SIDECAR_NAME).read_text())
    sidecar.update(sidecar_updates or {})
    (directory / FIT_SIDECAR_NAME).write_text(json.dumps(sidecar))
    return directory


def test_load_rejects_tampered_or_nonfinite_payload(tmp_path):
    fit = fit_pca(make_directions())
    source = tmp_path / "fit"
    save_fit(fit, source)

    tampered_identity = _copy_fit(
        tmp_path, "tampered-identity", source, sidecar_updates={"baseline_fit_identity": "0" * 64}
    )
    with pytest.raises(ValueError, match="identity"):
        load_fit(tampered_identity)

    tampered_width = _copy_fit(
        tmp_path, "tampered-width", source, sidecar_updates={"width": WIDTH + 1}
    )
    with pytest.raises(ValueError, match="shapes"):
        load_fit(tampered_width)

    tampered_payload = _copy_fit(
        tmp_path,
        "tampered-payload",
        source,
        tensors={"mean": fit.mean + 1, "basis": fit.basis},
    )
    with pytest.raises(ValueError, match="payload"):
        load_fit(tampered_payload)

    nonfinite = _copy_fit(
        tmp_path,
        "nonfinite",
        source,
        tensors={"mean": torch.full((WIDTH,), float("nan")), "basis": fit.basis},
    )
    sidecar_path = nonfinite / FIT_SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["tensors_sha256"] = sha256_file(nonfinite / FIT_TENSORS_NAME)
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="nonfinite"):
        load_fit(nonfinite)


def test_nonfinite_wrong_width_and_malformed_inputs_are_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        fit_pca(torch.full((4, WIDTH), float("nan")))
    poisoned = make_directions(n=4)
    poisoned[1, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        fit_pca(poisoned)
    with pytest.raises(ValueError, match="near-zero"):
        fit_pca(torch.zeros(2, WIDTH))
    with pytest.raises(ValueError, match="2D"):
        fit_pca(torch.zeros(2, 2, WIDTH))
    with pytest.raises(ValueError, match="at least one"):
        fit_pca([])
    with pytest.raises(ValueError, match="inconsistent"):
        fit_pca([torch.ones(WIDTH), torch.ones(WIDTH // 2)])
    fit = fit_pca(make_directions())
    with pytest.raises(ValueError, match="width"):
        reconstruct(fit, torch.ones(WIDTH // 2), text_bytes=8)
    with pytest.raises(ValueError, match="nonfinite"):
        reconstruct(fit, torch.full((WIDTH,), float("nan")), text_bytes=8)
    with pytest.raises(ValueError, match="nonnegative"):
        reconstruct(fit, torch.ones(WIDTH), text_bytes=-2)


def test_rank_zero_fit_falls_back_to_mean_direction(tmp_path):
    fit = fit_pca(torch.ones(8, WIDTH))
    assert fit.fitted_rank == 0
    direction, accounting = reconstruct(fit, torch.randn(WIDTH), text_bytes=10_000)
    assert accounting["rank_used"] == 0
    assert accounting["coefficient_bytes"] == 0
    assert torch.equal(direction, unit_direction(torch.ones(WIDTH)))
    save_fit(fit, tmp_path / "fit0")
    assert load_fit(tmp_path / "fit0").identity == fit.identity
