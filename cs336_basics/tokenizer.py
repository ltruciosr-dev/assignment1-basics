import regex as re
import json


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath) as f:
            vocab = json.load(f)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))

        return cls(vocab, merges, special_tokens)

    def _split_on_tokens(self, text: str, tokens: list[str] | None = None) -> list[str]:
        """Split text on multiple tokens efficiently."""
        if not tokens:
            return [text]

        # Escape and combine all tokens into one pattern
        escaped_tokens = [re.escape(token) for token in tokens]
        pattern = "|".join(escaped_tokens)

        # Split on any of the tokens
        return [chunk for chunk in re.split(pattern, text) if chunk]

    def _pre_tokenize(
        self,
        text: str,
        encoding: str = "utf-8",
    ) -> list[bytes]:
        """Pre tokenize a text and represent each pre-token as a sequence of bytes."""
        pre_tokens: list[bytes] = []
        if self.special_tokens:
            chunks = self._split_on_tokens(text, tokens=self.special_tokens)

        for chunk in chunks:
            for match in re.finditer(self.PAT, chunk):
                word: str = match.group(0)
                pre_tokens.append(word.encode(encoding))

        return pre_tokens

    def _apply_merges(self, pre_tokens: list[bytes]) -> list[int]:
        pass

    def encode(self, text: str):
        pre_tokens = self._pre_tokenize(text)
        tokens = self._apply_merges(pre_tokens)
        return tokens
