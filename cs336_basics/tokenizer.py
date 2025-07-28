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
        self.inv_vocab = self._get_inv_vocab()

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

    def _get_inv_vocab(self) -> dict[bytes, int]:
        inv_vocab = {}
        for k, v in self.vocab.items():
            inv_vocab[v] = k
        return inv_vocab

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
    ) -> list[list[int]]:
        """Pre tokenize a text and represent each pre-token as a sequence of bytes."""
        pre_tokens: list[list[int]] = []
        if self.special_tokens:
            chunks = self._split_on_tokens(text, tokens=self.special_tokens)
        else:
            chunks = [text]

        for chunk in chunks:
            for match in re.finditer(self.PAT, chunk):
                word: str = match.group(0)
                tokens: list[int] = [self.inv_vocab[c.encode(encoding)] for c in word]
                pre_tokens.append(tokens)

        return pre_tokens

    def _apply_merges(self, pre_tokens: list[bytes]) -> list[int]:
        """Merge the bytes based on the pre-trained merges and vocab"""
        # Compute all pairs
        vocab_size = len(self.vocab)
        for w_tokens in pre_tokens:
            prev_tokens = w_tokens
            # new_tokens = []
            w_valid = True
            breakpoint()
            while w_valid:
                # Get best pair
                pairs = []
                for w_pair in zip(prev_tokens, prev_tokens[1:]):
                    w_byte_pair = self.vocab[w_pair[0]] + self.vocab[w_pair[1]]
                    k_pair = self.inv_vocab.get(w_byte_pair)
                    if not k_pair:
                        k_pair = vocab_size + 1
                    pairs.append(k_pair)
                # Validate best pair
                if vocab_size < min(pairs):
                    w_valid = False

                breakpoint()
                # else: # reduce the size
                #     k = 0
                #     while k < len(prev_tokens) - 1:

                #         if (prev_tokens[k], )

        # for w_tokens in pre_tokens:
        #     partial_w_tokens = w_tokens
        #     for merge_byte_pair in self.merges:
        #         merge_pair = merge_byte_pair[0] + merge_byte_pair[1]
        #         k_merge = self.inv_vocab[merge_pair]
        #         new_w_tokens: list = []
        #         k = 0
        #         while k < len(partial_w_tokens) - 1:
        #             word_byte_pair = (
        #                 self.vocab[partial_w_tokens[k]],
        #                 self.vocab[partial_w_tokens[k+1]]
        #             )
        #             # breakpoint()
        #             if word_byte_pair == merge_byte_pair:
        #                 new_w_tokens.append(k_merge)
        #                 k += 2
        #             else:
        #                 new_w_tokens.append(w_tokens[k])

    def encode(self, text: str):
        text = "hello world"
        pre_tokens = self._pre_tokenize(text)
        tokens = self._apply_merges(pre_tokens)
        breakpoint()
        return tokens
