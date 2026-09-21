"""Tests for the fingerprint -> SMILES de novo Transformer decoder (MolForge).

Skipped entirely when torch is absent, since it is an optional extra (same
pattern as test_models.py). Deliberately tiny model/vocab: the goal is to
verify shapes, that beam search is bounded and deterministic, and that
training loss decreases on a trivial corpus — not to reach useful accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the optional 'train' extra")

from casmi.models.denovo import (  # noqa: E402
    DenovoConfig,
    FingerprintToSmilesDecoder,
    beam_search_generate,
    fingerprint_to_bit_sequence,
)
from casmi.models.smiles_tokenizer import SmilesVocabulary  # noqa: E402
from casmi.models.train_denovo import (  # noqa: E402
    DenovoTrainingConfig,
    build_denovo_training_rows,
    collate_denovo_batch,
    evaluate_denovo,
    load_denovo_checkpoint,
    save_denovo_checkpoint,
    train_denovo,
)

N_BITS = 32
MAX_FP_BITS = 16
CORPUS = ["CCO", "CCC", "c1ccccc1", "CCN", "CC(=O)O"]


@pytest.fixture
def vocabulary():
    return SmilesVocabulary.from_smiles_corpus(CORPUS)


@pytest.fixture
def config(vocabulary):
    return DenovoConfig(
        n_bits=N_BITS,
        vocab_size=vocabulary.size,
        d_model=16,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ffn_dim=32,
        max_length=20,
        max_fingerprint_bits=MAX_FP_BITS,
    )


@pytest.fixture
def model(config):
    return FingerprintToSmilesDecoder(config)


def make_fingerprint(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=N_BITS).astype(np.float32)


class TestFingerprintToBitSequence:
    def test_returns_ascending_set_bit_indices(self):
        fp = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
        seq = fingerprint_to_bit_sequence(fp, max_bits=10)
        np.testing.assert_array_equal(seq, [1, 3, 4])

    def test_empty_when_no_bits_set(self):
        fp = np.zeros(8, dtype=np.float32)
        seq = fingerprint_to_bit_sequence(fp, max_bits=10)
        assert seq.size == 0

    def test_truncates_to_max_bits(self):
        fp = np.ones(20, dtype=np.float32)
        seq = fingerprint_to_bit_sequence(fp, max_bits=5)
        assert seq.size == 5

    def test_accepts_continuous_probabilities_with_threshold(self):
        fp = np.array([0.9, 0.1, 0.6, 0.4], dtype=np.float32)
        seq = fingerprint_to_bit_sequence(fp, max_bits=10, threshold=0.5)
        np.testing.assert_array_equal(seq, [0, 2])


class TestFingerprintToSmilesDecoder:
    def test_encode_shape(self, model, config):
        batch, n_bits = 3, 5
        bit_indices = torch.zeros(batch, n_bits, dtype=torch.long)
        padding_mask = torch.zeros(batch, n_bits, dtype=torch.bool)
        memory = model.encode(bit_indices, padding_mask)
        assert memory.shape == (batch, n_bits, config.d_model)

    def test_forward_shape(self, model, config, vocabulary):
        batch, n_bits, length = 2, 5, 6
        bit_indices = torch.zeros(batch, n_bits, dtype=torch.long)
        bit_padding_mask = torch.zeros(batch, n_bits, dtype=torch.bool)
        tokens = torch.zeros(batch, length, dtype=torch.long)
        logits = model(bit_indices, bit_padding_mask, tokens)
        assert logits.shape == (batch, length, vocabulary.size)

    def test_decode_uses_causal_masking(self, model, config):
        """Logits at position t must not change when future tokens change."""
        model.eval()  # disable dropout so this isolates the causal-mask property
        batch, n_bits, length = 1, 4, 5
        bit_indices = torch.randint(0, config.n_bits, (batch, n_bits))
        bit_padding_mask = torch.zeros(batch, n_bits, dtype=torch.bool)
        memory = model.encode(bit_indices, bit_padding_mask)

        tokens_a = torch.zeros(batch, length, dtype=torch.long)
        tokens_b = tokens_a.clone()
        tokens_b[:, -1] = 1  # change only the last token

        logits_a = model.decode(tokens_a, memory, bit_padding_mask)
        logits_b = model.decode(tokens_b, memory, bit_padding_mask)
        # Earlier positions must be unaffected by a change to the last token.
        assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)

    def test_different_fingerprints_give_different_memory(self, model, config):
        bit_indices = torch.stack(
            [torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]
        )
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)
        memory = model.encode(bit_indices, padding_mask)
        assert not torch.allclose(memory[0], memory[1])

    def test_padding_mask_ignored_positions_dont_affect_others(self, model, config):
        """Padded bit positions must not leak information via attention."""
        bit_indices = torch.tensor([[1, 2, 0]], dtype=torch.long)
        mask_no_pad = torch.zeros(1, 3, dtype=torch.bool)
        mask_with_pad = torch.tensor([[False, False, True]])

        memory_no_pad = model.encode(bit_indices, mask_no_pad)
        memory_with_pad = model.encode(bit_indices, mask_with_pad)
        # Unpadded positions' encoding differs depending on whether the third
        # position participates in attention.
        assert not torch.allclose(memory_no_pad[:, :2], memory_with_pad[:, :2], atol=1e-6)


class TestBeamSearchGenerate:
    def test_returns_bounded_number_of_results(self, model, vocabulary):
        fp = make_fingerprint(0)
        results = beam_search_generate(model, fp, vocabulary, beam_width=4, max_length=10)
        assert len(results) <= 4

    def test_results_are_sorted_best_first(self, model, vocabulary):
        fp = make_fingerprint(1)
        results = beam_search_generate(model, fp, vocabulary, beam_width=6, max_length=10)
        log_probs = [lp for _, lp in results]
        assert log_probs == sorted(log_probs, reverse=True)

    def test_deterministic_for_same_input(self, model, vocabulary):
        fp = make_fingerprint(2)
        a = beam_search_generate(model, fp, vocabulary, beam_width=4, max_length=10)
        b = beam_search_generate(model, fp, vocabulary, beam_width=4, max_length=10)
        assert a == b

    def test_no_duplicate_smiles_in_output(self, model, vocabulary):
        fp = make_fingerprint(3)
        results = beam_search_generate(model, fp, vocabulary, beam_width=8, max_length=15)
        smiles = [s for s, _ in results]
        assert len(smiles) == len(set(smiles))

    def test_respects_max_length_bound(self, model, vocabulary):
        """Even an untrained model producing junk must never exceed the cap."""
        fp = make_fingerprint(4)
        results = beam_search_generate(model, fp, vocabulary, beam_width=3, max_length=8)
        for smiles, _ in results:
            # decoded length <= tokens generated <= max_length
            assert len(smiles) <= 8 * 4  # generous: longest single token is a bracket atom

    def test_empty_fingerprint_returns_no_results(self, model, vocabulary):
        fp = np.zeros(N_BITS, dtype=np.float32)
        results = beam_search_generate(model, fp, vocabulary, beam_width=4, max_length=10)
        assert results == []


class TestBuildDenovoTrainingRows:
    def test_builds_one_row_per_valid_structure(self, vocabulary):
        def fp_of(smiles):
            return make_fingerprint(hash(smiles) % 1000)

        rows = build_denovo_training_rows(CORPUS, fp_of, vocabulary, max_length=20)
        assert len(rows) == len(CORPUS)

    def test_skips_structures_with_no_fingerprint(self, vocabulary):
        def fp_of(smiles):
            return None if smiles == "CCC" else make_fingerprint(0)

        rows = build_denovo_training_rows(CORPUS, fp_of, vocabulary, max_length=20)
        assert len(rows) == len(CORPUS) - 1

    def test_respects_max_rows(self, vocabulary):
        rows = build_denovo_training_rows(
            CORPUS, lambda s: make_fingerprint(0), vocabulary, max_length=20, max_rows=2
        )
        assert len(rows) == 2


class TestCollateDenovoBatch:
    def test_shapes(self, vocabulary):
        rows = build_denovo_training_rows(
            CORPUS, lambda s: make_fingerprint(0), vocabulary, max_length=20
        )
        bit_indices, bit_padding_mask, inputs, targets = collate_denovo_batch(
            rows, pad_index=vocabulary.pad_index, max_fingerprint_bits=MAX_FP_BITS
        )
        assert bit_indices.shape[0] == len(rows)
        assert bit_padding_mask.shape == bit_indices.shape
        assert inputs.shape == targets.shape
        assert inputs.shape[0] == len(rows)

    def test_bit_flip_rate_zero_is_unchanged(self, vocabulary):
        rows = build_denovo_training_rows(
            CORPUS, lambda s: make_fingerprint(0), vocabulary, max_length=20
        )
        bit_indices, mask, _, _ = collate_denovo_batch(
            rows,
            pad_index=vocabulary.pad_index,
            max_fingerprint_bits=MAX_FP_BITS,
            bit_flip_rate=0.0,
        )
        expected_seq = fingerprint_to_bit_sequence(rows[0].fingerprint, max_bits=MAX_FP_BITS)
        n_valid = int((~mask[0]).sum().item())
        np.testing.assert_array_equal(bit_indices[0, :n_valid].numpy(), expected_seq)

    def test_bit_flip_rate_one_changes_bit_sequence(self, vocabulary):
        rows = build_denovo_training_rows(
            CORPUS, lambda s: make_fingerprint(0), vocabulary, max_length=20
        )
        rng = np.random.default_rng(0)
        bit_indices, mask, _, _ = collate_denovo_batch(
            rows,
            pad_index=vocabulary.pad_index,
            max_fingerprint_bits=MAX_FP_BITS,
            bit_flip_rate=1.0,
            rng=rng,
        )
        # Flipping every bit inverts which positions are "on"; with the same
        # deterministic fingerprint for every row, the flipped sequence must
        # differ from the unflipped one.
        unflipped_seq = fingerprint_to_bit_sequence(rows[0].fingerprint, max_bits=MAX_FP_BITS)
        n_valid = int((~mask[0]).sum().item())
        flipped_seq = bit_indices[0, :n_valid].numpy()
        assert not np.array_equal(flipped_seq, unflipped_seq)


class TestTrainingLoop:
    def test_loss_decreases(self, vocabulary):
        config = DenovoConfig(
            n_bits=N_BITS,
            vocab_size=vocabulary.size,
            d_model=16,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            ffn_dim=32,
            max_length=20,
            max_fingerprint_bits=MAX_FP_BITS,
        )
        # Deterministic per-structure fingerprints so the mapping is learnable.
        fp_map = {smiles: make_fingerprint(i) for i, smiles in enumerate(CORPUS)}
        rows = build_denovo_training_rows(
            CORPUS * 20, lambda s: fp_map[s], vocabulary, max_length=20
        )
        training_config = DenovoTrainingConfig(
            batch_size=8, max_steps=80, warmup_steps=5, log_every=5, bit_flip_rate=0.0
        )
        history = train_denovo(rows, config, vocabulary, training_config, device="cpu")
        assert history["loss"][-1] < history["loss"][0]

    def test_evaluate_denovo_returns_finite_loss(self, vocabulary):
        config = DenovoConfig(
            n_bits=N_BITS,
            vocab_size=vocabulary.size,
            d_model=16,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            ffn_dim=32,
            max_length=20,
            max_fingerprint_bits=MAX_FP_BITS,
        )
        model = FingerprintToSmilesDecoder(config)
        rows = build_denovo_training_rows(
            CORPUS, lambda s: make_fingerprint(0), vocabulary, max_length=20
        )
        loss = evaluate_denovo(
            model, rows, vocabulary.pad_index, device="cpu", max_fingerprint_bits=MAX_FP_BITS
        )
        assert np.isfinite(loss)


class TestCheckpointRoundtrip:
    def test_save_and_load_preserves_predictions(self, model, config, vocabulary, tmp_path):
        path = tmp_path / "denovo.pt"
        save_denovo_checkpoint(model, config, vocabulary, step=0, path=path)
        loaded_model, loaded_config, loaded_vocab = load_denovo_checkpoint(path, device="cpu")

        assert loaded_config == config
        assert loaded_vocab.index_to_token == vocabulary.index_to_token

        fp = make_fingerprint(0)
        a = beam_search_generate(model, fp, vocabulary, beam_width=3, max_length=10)
        b = beam_search_generate(loaded_model, fp, loaded_vocab, beam_width=3, max_length=10)
        assert a == b
