"""Tests for FPNet and its training loop.

Skipped entirely when torch is absent, since it is an optional extra. Runs on
CPU with a deliberately tiny model: the goal is to verify shapes, masking,
the Bayes scoring identity and that loss actually decreases — not to reach
useful accuracy.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the optional 'train' extra")

from casmi.models.fpnet import (  # noqa: E402
    ADDUCT_INDEX,
    FPNet,
    FPNetConfig,
    SinusoidalMass,
    collate_spectra,
    instrument_family,
    normalised_score,
    prepare_peaks,
    score_candidates,
)
from casmi.models.train import (  # noqa: E402
    TrainingConfig,
    TrainingRow,
    build_training_rows,
    compute_bit_weights,
    evaluate_fpnet,
    load_checkpoint,
    predict_logits,
    save_checkpoint,
    train_fpnet,
)
from tests.conftest import MOLECULES, build_library_table  # noqa: E402

N_BITS = 64


@pytest.fixture
def config():
    """Tiny architecture so tests run in seconds on CPU."""
    return FPNetConfig(n_bits=N_BITS, d_model=32, n_layers=2, n_heads=2, max_peaks=16)


@pytest.fixture
def model(config):
    return FPNet(config)


def make_batch(config, batch_size=3, n_peaks=8, device="cpu"):
    prepared = []
    rng = np.random.default_rng(0)
    for _ in range(batch_size):
        mz = np.sort(rng.uniform(50, 190, n_peaks)).astype(np.float32)
        intensity = rng.uniform(0.1, 1.0, n_peaks).astype(np.float32)
        prepared.append((mz, intensity))
    return collate_spectra(
        prepared=prepared,
        precursor_mz=[200.0] * batch_size,
        adducts=["[M+H]+"] * batch_size,
        instruments=["timsTOF"] * batch_size,
        collision_energies=[20.0] * batch_size,
        ionization_modes=["positive"] * batch_size,
        device=device,
    )


class TestInstrumentFamily:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("timsTOF", 0),
            ("Bruker timsTOF Pro", 0),
            ("Orbitrap Fusion", 1),
            ("Q Exactive", 1),
            ("LC-ESI-QTOF", 2),
            ("Ion Trap", 3),
            ("something else", 4),
            (None, 4),
        ],
    )
    def test_buckets(self, text, expected):
        assert instrument_family(text) == expected


class TestSinusoidalMass:
    def test_output_shape(self):
        emb = SinusoidalMass(16)
        assert emb(torch.tensor([100.0, 200.0])).shape == (2, 16)

    def test_distinguishes_close_masses(self):
        """0.01 Da separates real fragment formulas and must be visible."""
        emb = SinusoidalMass(64)
        a = emb(torch.tensor([100.00]))
        b = emb(torch.tensor([100.01]))
        assert not torch.allclose(a, b, atol=1e-6)

    def test_odd_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            SinusoidalMass(15)


class TestFPNetForward:
    def test_output_shape(self, model, config):
        logits = model(**make_batch(config))
        assert logits.shape == (3, N_BITS)

    def test_finite_outputs(self, model, config):
        assert torch.isfinite(model(**make_batch(config))).all()

    def test_padding_does_not_change_result(self, model, config):
        """Padded slots must be fully masked out of attention."""
        model.eval()
        mz = np.array([100.0, 150.0], dtype=np.float32)
        intensity = np.array([1.0, 0.5], dtype=np.float32)

        short = collate_spectra(
            [(mz, intensity)], [200.0], ["[M+H]+"], ["timsTOF"], [20.0], ["positive"]
        )
        padded_mz = np.concatenate([mz, np.zeros(6, np.float32)])
        padded_int = np.concatenate([intensity, np.zeros(6, np.float32)])
        padded = collate_spectra(
            [(padded_mz, padded_int)], [200.0], ["[M+H]+"], ["timsTOF"], [20.0], ["positive"]
        )
        # Mark the appended slots as padding, matching how collate would.
        padded["pad_mask"][0, 2:] = True

        with torch.no_grad():
            assert torch.allclose(model(**short), model(**padded), atol=1e-5)

    def test_different_spectra_give_different_logits(self, model, config):
        model.eval()
        a = collate_spectra(
            [(np.array([100.0], np.float32), np.array([1.0], np.float32))],
            [200.0], ["[M+H]+"], ["timsTOF"], [20.0], ["positive"],
        )
        b = collate_spectra(
            [(np.array([180.0], np.float32), np.array([1.0], np.float32))],
            [200.0], ["[M+H]+"], ["timsTOF"], [20.0], ["positive"],
        )
        with torch.no_grad():
            assert not torch.allclose(model(**a), model(**b), atol=1e-4)

    def test_adduct_affects_output(self, model, config):
        model.eval()
        peaks = [(np.array([100.0], np.float32), np.array([1.0], np.float32))]
        pos = collate_spectra(peaks, [200.0], ["[M+H]+"], ["timsTOF"], [20.0], ["positive"])
        neg = collate_spectra(peaks, [200.0], ["[M-H]-"], ["timsTOF"], [20.0], ["negative"])
        with torch.no_grad():
            assert not torch.allclose(model(**pos), model(**neg), atol=1e-4)

    def test_backward_produces_gradients(self, model, config):
        logits = model(**make_batch(config))
        logits.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


class TestPreparePeaks:
    def test_sorted_and_scaled(self, config):
        mz, intensity = prepare_peaks(
            np.array([180.0, 100.0, 150.0]), np.array([0.5, 1.0, 0.25]), 200.0, config
        )
        assert np.all(np.diff(mz) > 0)
        assert intensity.max() == pytest.approx(1.0)

    def test_drops_peaks_above_precursor(self, config):
        mz, _ = prepare_peaks(
            np.array([100.0, 500.0]), np.array([1.0, 1.0]), 200.0, config
        )
        assert 500.0 not in mz

    def test_respects_max_peaks(self):
        cfg = FPNetConfig(n_bits=N_BITS, max_peaks=5, peaks_per_window=2)
        rng = np.random.default_rng(0)
        mz = np.sort(rng.uniform(50, 400, 100))
        intensity = rng.uniform(0.01, 1.0, 100)
        out_mz, _ = prepare_peaks(mz, intensity, 500.0, cfg)
        assert len(out_mz) == 5

    def test_preserves_low_mass_peaks_via_windowing(self):
        """Windowed selection must keep a weak low-mass peak a global top-N drops."""
        cfg = FPNetConfig(n_bits=N_BITS, max_peaks=4, peaks_per_window=1, window_da=50.0)
        mz = np.array([60.0, 210.0, 220.0, 230.0, 240.0])
        intensity = np.array([0.05, 1.0, 0.99, 0.98, 0.97])
        out_mz, _ = prepare_peaks(mz, intensity, 300.0, cfg)
        assert 60.0 in out_mz

    def test_empty_input(self, config):
        mz, intensity = prepare_peaks(np.array([]), np.array([]), 200.0, config)
        assert mz.size == 0 and intensity.size == 0

    def test_all_zero_intensity(self, config):
        mz, _ = prepare_peaks(np.array([100.0]), np.array([0.0]), 200.0, config)
        assert mz.size == 0


class TestCollate:
    def test_pad_mask_marks_short_rows(self, config):
        prepared = [
            (np.array([100.0], np.float32), np.array([1.0], np.float32)),
            (np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32)),
        ]
        batch = collate_spectra(
            prepared, [200.0, 200.0], ["[M+H]+"] * 2, ["timsTOF"] * 2, [20.0] * 2,
            ["positive"] * 2,
        )
        assert batch["pad_mask"][0, 1].item() is True
        assert batch["pad_mask"][1, 1].item() is False

    def test_unknown_adduct_maps_to_unk(self, config):
        batch = collate_spectra(
            [(np.array([100.0], np.float32), np.array([1.0], np.float32))],
            [200.0], ["[M+Unobtainium]+"], ["timsTOF"], [20.0], ["positive"],
        )
        assert batch["adduct"][0].item() == ADDUCT_INDEX["<unk>"]

    def test_nan_collision_energy_defaulted(self, config):
        batch = collate_spectra(
            [(np.array([100.0], np.float32), np.array([1.0], np.float32))],
            [200.0], ["[M+H]+"], ["timsTOF"], [float("nan")], ["positive"],
        )
        assert torch.isfinite(batch["collision_energy"]).all()

    def test_empty_batch_rejected(self):
        with pytest.raises(ValueError, match="empty batch"):
            collate_spectra([], [], [], [], [], [])


class TestBayesScoring:
    def test_equivalent_to_full_log_likelihood_ranking(self):
        """The dot product must induce the same ranking as exact log-likelihood.

        This identity is the reason a full candidate pool can be scored with one
        matmul, so it is worth asserting rather than assuming.
        """
        rng = np.random.default_rng(0)
        n_bits, n_candidates = 32, 25
        logits = rng.normal(0, 2, n_bits).astype(np.float32)
        fps = rng.integers(0, 2, (n_candidates, n_bits)).astype(np.uint8)

        fast = score_candidates(fps, logits)
        # Exact: sum_i [f_i * z_i - log(1 + exp(z_i))]
        exact = (fps.astype(np.float32) * logits).sum(1) - np.log1p(np.exp(logits)).sum()

        assert np.allclose(np.argsort(-fast), np.argsort(-exact))
        assert np.allclose(fast - exact, fast[0] - exact[0], atol=1e-3)

    def test_matching_fingerprint_scores_highest(self):
        """A candidate equal to the model's confident prediction must win."""
        n_bits = 32
        truth = np.zeros(n_bits, dtype=np.uint8)
        truth[:8] = 1
        logits = np.where(truth > 0, 5.0, -5.0).astype(np.float32)
        decoys = np.random.default_rng(1).integers(0, 2, (10, n_bits)).astype(np.uint8)
        fps = np.vstack([truth[None, :], decoys])
        assert int(np.argmax(score_candidates(fps, logits))) == 0

    def test_width_mismatch_raises(self):
        with pytest.raises(ValueError, match="width mismatch"):
            score_candidates(np.zeros((2, 8), np.uint8), np.zeros(16, np.float32))

    def test_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            score_candidates(np.zeros(8, np.uint8), np.zeros(8, np.float32))

    def test_normalised_penalises_bit_count(self):
        """Normalisation must offset the raw score's bias toward large molecules."""
        n_bits = 32
        logits = np.ones(n_bits, dtype=np.float32)
        few = np.zeros(n_bits, np.uint8)
        few[:4] = 1
        many = np.ones(n_bits, np.uint8)
        fps = np.vstack([few, many])
        raw = score_candidates(fps, logits)
        norm = normalised_score(fps, logits)
        assert raw[1] > raw[0]
        assert norm[1] / norm[0] < raw[1] / raw[0]


class TestBitWeights:
    def test_rare_bits_weighted_higher(self):
        """Rare bits get boosted; bits at or above 50% frequency sit at the floor."""
        fps = np.zeros((100, 4), dtype=np.uint8)
        fps[:, 0] = 1        # always set -> no information -> floor
        fps[:50, 1] = 1      # half -> balanced -> floor
        fps[:2, 2] = 1       # rare -> boosted
        fps[:10, 3] = 1      # uncommon -> boosted, but less than the rare bit
        weights = compute_bit_weights(fps)
        assert weights[2] > weights[3] > weights[1]
        assert weights[1] == pytest.approx(1.0)
        assert weights[0] == pytest.approx(1.0)

    def test_capped(self):
        fps = np.zeros((1000, 2), dtype=np.uint8)
        fps[0, 0] = 1  # extremely rare
        assert compute_bit_weights(fps, max_weight=5.0).max() <= 5.0

    def test_never_below_one(self):
        fps = np.ones((10, 3), dtype=np.uint8)
        assert compute_bit_weights(fps).min() >= 1.0

    def test_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_bit_weights(np.zeros(4, dtype=np.uint8))


def make_rows(n=24, n_bits=N_BITS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        n_peaks = 8
        mz = np.sort(rng.uniform(50, 190, n_peaks)).astype(np.float32)
        intensity = rng.uniform(0.2, 1.0, n_peaks).astype(np.float32)
        fingerprint = np.zeros(n_bits, dtype=np.uint8)
        # Two structural classes, so there is a learnable mapping.
        if i % 2 == 0:
            fingerprint[:16] = 1
            mz = mz * 0.5 + 50.0
        else:
            fingerprint[16:32] = 1
        rows.append(
            TrainingRow(
                mz=mz.astype(np.float32),
                intensity=intensity,
                precursor_mz=200.0,
                adduct="[M+H]+",
                instrument="timsTOF",
                collision_energy=20.0,
                ionization_mode="positive",
                fingerprint=fingerprint,
            )
        )
    return rows


class TestBuildTrainingRows:
    def test_builds_rows_with_fingerprints(self, config):
        table = build_library_table([(n, s, i) for i, (n, s) in enumerate(MOLECULES[:4])])
        fingerprints = {}

        def lookup(key):
            fingerprints.setdefault(key, np.ones(N_BITS, dtype=np.uint8))
            return fingerprints[key]

        rows = build_training_rows(table, lookup, config)
        assert len(rows) == 4
        assert all(r.fingerprint.shape == (N_BITS,) for r in rows)

    def test_row_mask_restricts_to_train_side(self, config):
        """Omitting the mask would train on held-out structures."""
        table = build_library_table([(n, s, i) for i, (n, s) in enumerate(MOLECULES[:4])])
        mask = np.array([True, False, True, False])
        rows = build_training_rows(
            table, lambda k: np.ones(N_BITS, dtype=np.uint8), config, row_mask=mask
        )
        assert len(rows) == 2

    def test_skips_structures_without_fingerprints(self, config):
        table = build_library_table([(n, s, i) for i, (n, s) in enumerate(MOLECULES[:4])])
        assert build_training_rows(table, lambda k: None, config) == []

    def test_wrong_mask_size_raises(self, config):
        table = build_library_table([("phenol", MOLECULES[1][1], 0)])
        with pytest.raises(ValueError, match="!="):
            build_training_rows(
                table, lambda k: np.ones(N_BITS, np.uint8), config,
                row_mask=np.ones(5, dtype=bool),
            )

    def test_max_rows_caps(self, config):
        table = build_library_table([(n, s, i) for i, (n, s) in enumerate(MOLECULES[:6])])
        rows = build_training_rows(
            table, lambda k: np.ones(N_BITS, np.uint8), config, max_rows=2
        )
        assert len(rows) == 2


class TestTrainingLoop:
    def test_runs_and_reduces_loss(self, config, tmp_path):
        """Short CPU run: loss must actually decrease, not merely execute."""
        rows = make_rows(n=32)
        history = train_fpnet(
            rows,
            config,
            TrainingConfig(
                batch_size=8,
                max_steps=40,
                warmup_steps=5,
                log_every=5,
                eval_every=0,
                checkpoint_every=0,
                use_amp=False,
            ),
            device="cpu",
            output_dir=tmp_path,
            multi_gpu=False,
        )
        assert history["loss"]
        assert history["loss"][-1] < history["loss"][0]

    def test_writes_final_checkpoint(self, config, tmp_path):
        train_fpnet(
            make_rows(n=16),
            config,
            TrainingConfig(
                batch_size=8, max_steps=4, warmup_steps=1, log_every=10,
                eval_every=0, checkpoint_every=0, use_amp=False,
            ),
            device="cpu",
            output_dir=tmp_path,
            multi_gpu=False,
        )
        assert (tmp_path / "fpnet_final.pt").exists()
        assert (tmp_path / "history.json").exists()

    def test_validation_metric_computed(self, config, tmp_path):
        history = train_fpnet(
            make_rows(n=16),
            config,
            TrainingConfig(
                batch_size=8, max_steps=6, warmup_steps=1, log_every=10,
                eval_every=2, checkpoint_every=0, use_amp=False,
            ),
            device="cpu",
            output_dir=tmp_path,
            validation_rows=make_rows(n=8, seed=1),
            multi_gpu=False,
        )
        assert history["val_cosine"]

    def test_empty_dataset_rejected(self, config, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            train_fpnet([], config, TrainingConfig(), device="cpu", output_dir=tmp_path)


class TestCheckpointRoundtrip:
    def test_save_load_preserves_outputs(self, config, tmp_path):
        model = FPNet(config)
        model.eval()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, config, step=7, path=path)

        loaded, loaded_config = load_checkpoint(path, device="cpu")
        assert loaded_config.n_bits == config.n_bits
        assert loaded_config.n_layers == config.n_layers

        batch = make_batch(config, batch_size=2)
        with torch.no_grad():
            assert torch.allclose(model(**batch), loaded(**batch), atol=1e-6)


class TestPredictLogits:
    def test_averages_across_spectra(self, config):
        model = FPNet(config)
        model.eval()
        spectra = [
            (np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32)),
            (np.array([110.0, 160.0], np.float32), np.array([1.0, 0.4], np.float32)),
        ]
        logits = predict_logits(
            model, spectra, [200.0, 200.0], ["[M+H]+"] * 2, ["timsTOF"] * 2,
            [20.0, 40.0], ["positive"] * 2, config,
        )
        assert logits.shape == (N_BITS,)
        assert np.isfinite(logits).all()

    def test_all_empty_spectra_returns_zeros(self, config):
        model = FPNet(config)
        logits = predict_logits(
            model, [(np.array([]), np.array([]))], [200.0], ["[M+H]+"],
            ["timsTOF"], [20.0], ["positive"], config,
        )
        assert logits.shape == (N_BITS,)
        assert np.all(logits == 0.0)


class TestEvaluateFPNet:
    def test_returns_cosine_in_range(self, config):
        model = FPNet(config)
        value = evaluate_fpnet(model, make_rows(n=8), device="cpu", batch_size=4)
        assert -1.0 <= value <= 1.0

    def test_empty_rows(self, config):
        assert evaluate_fpnet(FPNet(config), [], device="cpu") == 0.0
