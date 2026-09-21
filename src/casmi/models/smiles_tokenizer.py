"""Character-level SMILES tokenizer for the de novo decoder.

A fixed, small vocabulary built from the characters that actually occur in
SMILES strings (elements, bonds, ring-closure digits, branch parens, charge/
stereo marks) rather than a learned subword tokenizer — SMILES's alphabet is
small and mostly single characters, with a handful of two-character element
symbols (``Cl``, ``Br``) that must be matched greedily before falling back to
single characters, or every chlorine becomes a nonsense ``C`` + ``l`` pair.
"""

from __future__ import annotations

#: Two-character tokens that must be matched before single-character fallback.
#: Bracketed atoms (``[nH]``, ``[13C]`` etc.) are handled by the bracket scan
#: below instead, so bare elements are the only multi-character case here.
_TWO_CHAR_TOKENS: tuple[str, ...] = ("Cl", "Br")

#: Special control tokens. Index order is load-bearing (checkpoints store
#: vocab size, not the mapping itself, so this must never be reordered).
START = "<start>"
END = "<end>"
PAD = "<pad>"
UNK = "<unk>"


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into a list of atomic tokens.

    Bracketed atoms (e.g. ``[nH]``, ``[C@@H]``, ``[13C]``) are kept as one
    token each, since splitting their contents would make many distinct
    chemical meanings collide into the same character sequence.
    """
    tokens: list[str] = []
    i = 0
    n = len(smiles)
    while i < n:
        char = smiles[i]
        if char == "[":
            close = smiles.find("]", i)
            if close == -1:
                tokens.append(char)
                i += 1
                continue
            tokens.append(smiles[i : close + 1])
            i = close + 1
            continue
        two = smiles[i : i + 2]
        if two in _TWO_CHAR_TOKENS:
            tokens.append(two)
            i += 2
            continue
        tokens.append(char)
        i += 1
    return tokens


class SmilesVocabulary:
    """Fixed token <-> index mapping built once from a training corpus.

    Args:
        tokens: Unique tokens observed in the training corpus (excluding the
            special control tokens, which are always added).
    """

    def __init__(self, tokens: list[str]) -> None:
        specials = [PAD, START, END, UNK]
        ordered = specials + sorted(set(tokens) - set(specials))
        self.index_to_token = ordered
        self.token_to_index = {tok: i for i, tok in enumerate(ordered)}

    @property
    def size(self) -> int:
        return len(self.index_to_token)

    @property
    def pad_index(self) -> int:
        return self.token_to_index[PAD]

    @property
    def start_index(self) -> int:
        return self.token_to_index[START]

    @property
    def end_index(self) -> int:
        return self.token_to_index[END]

    def encode(self, smiles: str, max_length: int | None = None) -> list[int]:
        """Tokenize and map to indices, wrapped in start/end tokens.

        Truncates (keeping the end token) rather than raising when
        ``max_length`` is exceeded, since a handful of oversized training
        SMILES should not crash a whole training run.
        """
        unk = self.token_to_index[UNK]
        ids = [self.start_index]
        ids.extend(self.token_to_index.get(t, unk) for t in tokenize_smiles(smiles))
        ids.append(self.end_index)
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.end_index]
        return ids

    def decode(self, indices: list[int]) -> str:
        """Map indices back to a SMILES string, stopping at the first end token."""
        chars = []
        for idx in indices:
            if idx == self.end_index:
                break
            if idx in (self.pad_index, self.start_index):
                continue
            if 0 <= idx < len(self.index_to_token):
                token = self.index_to_token[idx]
                if token not in (PAD, START, END, UNK):
                    chars.append(token)
        return "".join(chars)

    @classmethod
    def from_smiles_corpus(cls, smiles_list: list[str]) -> SmilesVocabulary:
        """Build a vocabulary from every token appearing in a corpus."""
        seen: set[str] = set()
        for smiles in smiles_list:
            seen.update(tokenize_smiles(smiles))
        return cls(sorted(seen))

    def to_dict(self) -> dict:
        return {"tokens": self.index_to_token}

    @classmethod
    def from_dict(cls, payload: dict) -> SmilesVocabulary:
        tokens = payload["tokens"]
        vocab = cls.__new__(cls)
        vocab.index_to_token = list(tokens)
        vocab.token_to_index = {tok: i for i, tok in enumerate(vocab.index_to_token)}
        return vocab
