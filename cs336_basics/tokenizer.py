import regex as re
import json
import pickle
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import islice
from collections.abc import Iterable, Iterator


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_vocab = self._get_inv_vocab()

    @classmethod
    def from_pkl_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    @classmethod
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
        sorted_tokens = sorted(tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]
        pattern = "|".join(escaped_tokens)

        return [chunk for chunk in re.split(f"({pattern})", text) if chunk]

    def _pre_tokenize(
        self,
        text: str,
        encoding: str = "utf-8",
    ) -> list[list[int]]:
        """Optimized pre tokenize with better memory management."""
        pre_tokens: list[list[int]] = []

        if self.special_tokens:
            chunks = self._split_on_tokens(text, tokens=self.special_tokens)
            for chunk in chunks:
                if chunk in self.special_tokens:
                    special_bytes = chunk.encode(encoding)
                    special_token = self.inv_vocab.get(special_bytes)
                    if special_token is not None:
                        pre_tokens.append([special_token])
                    continue

                # Process chunk in batches for better memory usage
                for match in re.finditer(self.PAT, chunk):
                    word: str = match.group(0)
                    word_bytes = word.encode(encoding)
                    word_tokens = []
                    for i in range(len(word_bytes)):
                        token = self.inv_vocab.get(word_bytes[i : i + 1])
                        if token is not None:
                            word_tokens.append(token)
                    if word_tokens:
                        pre_tokens.append(word_tokens)
        else:
            for match in re.finditer(self.PAT, text):
                word: str = match.group(0)
                word_bytes = word.encode(encoding)
                word_tokens = []
                for i in range(len(word_bytes)):
                    token = self.inv_vocab.get(word_bytes[i : i + 1])
                    if token is not None:
                        word_tokens.append(token)
                if word_tokens:
                    pre_tokens.append(word_tokens)

        return pre_tokens

    def _apply_merges(self, pre_tokens: list[list[int]]) -> list[int]:
        """Optimized merge application with reduced redundant computations."""
        after_merge_tokens: list[int] = []
        vocab_size = len(self.vocab)

        for w_tokens in pre_tokens:
            if not w_tokens:
                continue

            new_tokens = w_tokens[:]

            while len(new_tokens) >= 2:
                # Pre-compute all possible pairs and their merged tokens
                pairs_info = []
                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i], new_tokens[i + 1])
                    w_byte_pair = self.vocab[pair[0]] + self.vocab[pair[1]]
                    k_pair = self.inv_vocab.get(w_byte_pair)
                    if k_pair is not None and k_pair < vocab_size:
                        pairs_info.append((i, k_pair))

                if not pairs_info:
                    break

                # Find the best pair (lowest token id)
                best_idx, best_token = min(pairs_info, key=lambda x: x[1])

                # Apply the merge
                merged_tokens = []
                i = 0
                while i < len(new_tokens):
                    if i == best_idx and i + 1 < len(new_tokens):
                        merged_tokens.append(best_token)
                        i += 2
                    else:
                        merged_tokens.append(new_tokens[i])
                        i += 1

                new_tokens = merged_tokens

            after_merge_tokens.extend(new_tokens)

        return after_merge_tokens

    def encode(self, text: str) -> list[int]:
        """Optimized encode function."""
        pre_tokens = self._pre_tokenize(text)
        tokens = self._apply_merges(pre_tokens)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Optimized decode function using join for better performance."""
        if not tokens:
            return ""

        # Use list comprehension and join for better performance
        text_bytes = b"".join(self.vocab[token] for token in tokens)
        return text_bytes.decode(errors="replace")

    def encode_iterable(self, iterable: Iterable[str], max_workers: int = None, batch_size: int = 200) -> Iterator[int]:
        """Memory-efficient streaming encode_iterable with controlled pipeline depth."""

        def batch_iterator(iterable, batch_size):
            """Create batches from iterable without loading everything into memory."""
            iterator = iter(iterable)
            while True:
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break
                yield batch

        # For sequential processing (no multiprocessing overhead)
        if max_workers == 1:
            for text in iterable:
                tokens = self.encode(text)
                yield from tokens
            return

        max_workers = max_workers or min(4, (os.cpu_count() or 1))
        pipeline_depth = 2 * max_workers

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with the tokenizer state
            encode_batch_func = partial(
                self._encode_batch_worker,
                vocab=self.vocab,
                inv_vocab=self.inv_vocab,
                special_tokens=self.special_tokens,
                pat=self.PAT,
            )

            # Pipeline processing with controlled memory usage
            batch_iter = batch_iterator(iterable, batch_size)
            futures = {}
            batches_processed = 0

            # Submit initial batches to keep workers busy (reduced pipeline)
            for _ in range(pipeline_depth):
                try:
                    batch = next(batch_iter)
                    future = executor.submit(encode_batch_func, batch)
                    futures[future] = None
                except StopIteration:
                    break

            # Process results as they complete and submit new batches
            while futures:
                # Wait for at least one future to complete
                for future in as_completed(futures):
                    # Yield results from completed batch immediately
                    batch_results = future.result()
                    for tokens in batch_results:
                        yield from tokens

                    # Clear batch results immediately to free memory
                    del batch_results

                    # Remove completed future
                    del futures[future]

                    # Periodic garbage collection to prevent memory buildup
                    batches_processed += 1
                    if batches_processed % 5000 == 0:  # Every 50 batches
                        gc.collect()

                    # Submit next batch if available
                    try:
                        batch = next(batch_iter)
                        new_future = executor.submit(encode_batch_func, batch)
                        futures[new_future] = None
                    except StopIteration:
                        pass

                    break  # Process one future at a time to maintain order

    @staticmethod
    def _encode_batch_worker(
        batch: list[str], vocab: dict[int, bytes], inv_vocab: dict[bytes, int], special_tokens: list[str], pat: str
    ) -> list[list[int]]:
        """Memory-efficient worker function for parallel batch processing."""
        # Create a temporary tokenizer instance for this worker
        temp_tokenizer = BPETokenizer.__new__(BPETokenizer)
        temp_tokenizer.vocab = vocab
        temp_tokenizer.inv_vocab = inv_vocab
        temp_tokenizer.special_tokens = special_tokens
        temp_tokenizer.PAT = pat

        # Process and yield results immediately to reduce memory usage
        batch_results = []
        for text in batch:
            tokens = temp_tokenizer.encode(text)
            batch_results.append(tokens)

            # Clear the text reference to help with memory cleanup
            text = None

        # Clear the batch to help with garbage collection
        batch.clear()
        return batch_results
