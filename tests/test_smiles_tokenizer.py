"""Tests for the character-level SMILES tokenizer.

No torch dependency, so this always runs (unlike the de novo model tests,
which require the optional 'train' extra).
"""

from __future__ import annotations

from casmi.models.smiles_tokenizer import (
    END,
    PAD,
    START,
    UNK,
    SmilesVocabulary,
    tokenize_smiles,
)


class TestTokenizeSmiles:
    def test_simple_atoms(self):
        assert tokenize_smiles("CCO") == ["C", "C", "O"]

    def test_two_character_elements(self):
        assert tokenize_smiles("ClCCBr") == ["Cl", "C", "C", "Br"]

    def test_bracketed_atoms_stay_whole(self):
        assert tokenize_smiles("c1cc[nH]cc1") == [
            "c", "1", "c", "c", "[nH]", "c", "c", "1",
        ]

    def test_stereo_and_isotope_brackets(self):
        tokens = tokenize_smiles("[13C]")
        assert tokens == ["[13C]"]
        tokens = tokenize_smiles("[C@@H]")
        assert tokens == ["[C@@H]"]

    def test_branches_and_bonds(self):
        assert tokenize_smiles("CC(=O)O") == ["C", "C", "(", "=", "O", ")", "O"]

    def test_unclosed_bracket_falls_back_to_single_char(self):
        # Malformed input must not crash tokenization.
        assert tokenize_smiles("C[") == ["C", "["]

    def test_empty_string(self):
        assert tokenize_smiles("") == []


class TestSmilesVocabulary:
    def test_specials_always_present(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO"])
        for special in (PAD, START, END, UNK):
            assert special in vocab.token_to_index

    def test_pad_index_is_zero(self):
        """Padding at index 0 is relied on by collate/loss masking elsewhere."""
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO", "c1ccccc1"])
        assert vocab.pad_index == 0

    def test_encode_wraps_with_start_and_end(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO"])
        ids = vocab.encode("CCO")
        assert ids[0] == vocab.start_index
        assert ids[-1] == vocab.end_index
        assert len(ids) == len(tokenize_smiles("CCO")) + 2

    def test_decode_stops_at_end_token(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO", "c1ccccc1"])
        ids = vocab.encode("CCO")
        assert vocab.decode(ids) == "CCO"

    def test_decode_ignores_padding_after_end(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO"])
        ids = vocab.encode("CCO") + [vocab.pad_index] * 5
        assert vocab.decode(ids) == "CCO"

    def test_roundtrip_preserves_two_char_elements(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["ClCCBr", "CCO"])
        ids = vocab.encode("ClCCBr")
        assert vocab.decode(ids) == "ClCCBr"

    def test_unknown_token_maps_to_unk_not_crash(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO"])
        # 'Br' was never seen during corpus construction.
        ids = vocab.encode("CCBr")
        assert vocab.token_to_index[UNK] in ids

    def test_truncates_to_max_length_keeping_end_token(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["c1ccccc1CCCCCCCCCC"])
        ids = vocab.encode("c1ccccc1CCCCCCCCCC", max_length=5)
        assert len(ids) == 5
        assert ids[-1] == vocab.end_index

    def test_to_dict_from_dict_roundtrip(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO", "c1ccccc1", "ClBr"])
        restored = SmilesVocabulary.from_dict(vocab.to_dict())
        assert restored.index_to_token == vocab.index_to_token
        assert restored.encode("CCO") == vocab.encode("CCO")

    def test_size_matches_token_count(self):
        vocab = SmilesVocabulary.from_smiles_corpus(["CCO"])
        assert vocab.size == len(vocab.index_to_token)
