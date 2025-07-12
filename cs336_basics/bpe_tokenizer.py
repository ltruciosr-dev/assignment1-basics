import os
import regex as re
import concurrent.futures
from tqdm import tqdm
from typing import BinaryIO
from collections import Counter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _initialize_vocabulary(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    """
    We initialize our map of vocabulary bytes into tokens, when the key could be
    actual bytes for special vocabulary.

    Args:
        special_tokens (list[str]): The set of strings to include as individual tokens
    Returns:
        dict[bytes, int]: The map between vocabulary and tokens
    """
    vocab = {}
    for i in range(2**8):
        vocab[i] = bytes([i])

    # Add special vocabulary
    if not special_tokens:
        return vocab

    for sv in special_tokens:
        vocab[len(vocab)] = bytes(sv, encoding="utf-8")
    return vocab


def _word_to_bytes(word: str) -> list[int]:
    """
    Given a word, return a list of bytes on utf-8 encoding.

    Args:
        word (str): The word to convert to bytes.

    Returns:
        list[int]: The list of bytes on utf-8 encoding.
    """
    return list(word.encode("utf-8"))


def _split_on_tokens(text: str, tokens: list[str]) -> list[str]:
    """Split text on multiple tokens efficiently."""
    if not tokens:
        return [text]

    # Escape and combine all tokens into one pattern
    escaped_tokens = [re.escape(token) for token in tokens]
    pattern = "|".join(escaped_tokens)

    # Split on any of the tokens
    return [chunk for chunk in re.split(pattern, text) if chunk]


def _pretokenizer_dict(text: str, special_tokens: list[str] = None) -> dict[str, int]:
    """
    Given a text, return a list of subwords bytes and their counts.

    Args:
        text (str): The text to pre-tokenize.

    Returns:
        list[list[int], int]: The list of subwords bytes and their counts.
    """
    pre_tokens: dict[str, int] = {}

    # Find the matches
    if special_tokens:
        chunks = _split_on_tokens(text, special_tokens)
    else:
        chunks = [text]

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word: str = match.group(0)
            if word not in pre_tokens:
                pre_tokens[word] = 1
            else:
                pre_tokens[word] += 1

    return pre_tokens


def _pretokenizer(text: str, special_tokens: list[str] = None) -> tuple[list[int], int]:
    """
    Given a text, return a list of subwords bytes and their counts.

    Args:
        text (str): The text to pre-tokenize.

    Returns:
        list[list[int], int]: The list of subwords bytes and their counts.
    """
    pre_tokens: dict[str, int] = {}

    # Find the matches
    if special_tokens:
        chunks = _split_on_tokens(text, special_tokens)
    else:
        chunks = [text]

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word: str = match.group(0)
            if word not in pre_tokens:
                pre_tokens[word] = 1
            else:
                pre_tokens[word] += 1
    return [[_word_to_bytes(k), v] for k, v in pre_tokens.items()]


def _single_bpe_merge(pre_tokens: dict[list[int], int], vocab: dict[bytes, int], merges: list[tuple[bytes, bytes]]):
    """
    Get the most frequent pair from the pre_tokens, apply the changes and update the vocabulary.
    """
    # 1. Get the more frequent pair
    pair_counter: dict[tuple[int], int] = {}
    for i in range(len(pre_tokens)):
        token = pre_tokens[i][0]
        count = pre_tokens[i][1]
        if len(token) < 2:
            continue
        for a, b in zip(token, token[1:]):
            if (a, b) in pair_counter.keys():
                pair_counter[(a, b)] += count
            else:
                pair_counter[(a, b)] = count

    if not pair_counter:
        return False

    # 2. Append merge into vocabulary and merges
    idx = len(vocab)
    frequent_pair: tuple[int] = max(pair_counter, key=lambda k: (pair_counter[k], (vocab[k[0]], vocab[k[1]])))
    vocab[idx] = b"".join([vocab[token] for token in frequent_pair])
    merges.append((vocab[frequent_pair[0]], vocab[frequent_pair[1]]))

    # 3. Modify the pre_tokens accordingly
    for i in range(len(pre_tokens)):
        token: list = pre_tokens[i][0]
        new_token: list = []
        if len(token) < 2:
            continue
        else:
            k = 0
            while k < len(token) - 1:
                if (token[k], token[k + 1]) == frequent_pair:
                    new_token.append(idx)
                    k += 2
                else:
                    new_token.append(token[k])
                    k += 1
            if k < len(token):
                new_token.append(token[-1])
        pre_tokens[i][0] = new_token

    return True


def _find_first_occurrence_optimized(data: bytes, patterns: list[bytes]) -> int:
    """Find the position of the first occurrence of any pattern in the list."""
    min_pos = len(data)
    found = False

    if len(patterns) == 0:
        patterns = [b"\n"]

    for pattern in patterns:
        pos = data.find(pattern)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            found = True
            # Early exit if we found a match at position 0
            if pos == 0:
                break

    return min_pos if found else -1


def _find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, special_tokens: list[str] | None) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    b_special_tokens = [token.encode("utf-8") for token in special_tokens]
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = _find_first_occurrence_optimized(mini_chunk, b_special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _merge_tokens(
    pre_tokens: dict[list[int], int], special_tokens: list[str], vocab_size: int
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """
    Merge tokens until the vocabulary size is reached.

    Args:
        pre_tokens (dict[list[int], int]): The pre-tokens with their counts.
        vocab (dict[bytes, int]): The vocabulary.
        merges (list[tuple[bytes, bytes]]): The merges.
        vocab_size (int): The vocabulary size.
    """
    merges = []
    vocab = _initialize_vocabulary(special_tokens)

    # Iterate until vocab size reached or no merges are available
    has_merge = True
    while len(vocab) < vocab_size and has_merge:
        has_merge = _single_bpe_merge(pre_tokens, vocab, merges)

    return vocab, merges


def _merge_tokens_optimized(
    pre_tokens: dict[list[int], int], special_tokens: list[str], vocab_size: int
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """
    Merge tokens until the vocabulary size is reached.

    Args:
        pre_tokens (dict[list[int], int]): The pre-tokens with their counts.
        vocab (dict[bytes, int]): The vocabulary.
        merges (list[tuple[bytes, bytes]]): The merges.
        vocab_size (int): The vocabulary size.
    """
    has_merge = True
    merges = []
    vocab = _initialize_vocabulary(special_tokens)

    # 1. Initialize the pair_counter for all the vocabulary
    pair_counter: dict[tuple[int], int] = {}
    for i in range(len(pre_tokens)):
        w_tokens = pre_tokens[i][0]
        count = pre_tokens[i][1]
        if len(w_tokens) < 2:
            continue
        for a, b in zip(w_tokens, w_tokens[1:]):
            if (a, b) in pair_counter.keys():
                pair_counter[(a, b)] += count
            else:
                pair_counter[(a, b)] = count

    if not pair_counter:
        has_merge = False

    pbar = tqdm(total=vocab_size - len(vocab), desc="BPE merges", unit="merge")
    while len(vocab) < vocab_size and has_merge:
        # Find the most frequent pair
        k_vocab = len(vocab)
        frequent_pair: tuple[int] = max(pair_counter, key=lambda k: (pair_counter[k], (vocab[k[0]], vocab[k[1]])))
        vocab[k_vocab] = b"".join([vocab[token] for token in frequent_pair])  # update vocab
        merges.append((vocab[frequent_pair[0]], vocab[frequent_pair[1]]))  # update merges

        # Modify the pre_tokens accordingly
        for i in range(len(pre_tokens)):
            w_tokens: list = pre_tokens[i][0]
            new_w_tokens: list = []
            count = pre_tokens[i][1]
            if len(w_tokens) < 2:
                continue
            else:
                # Update the word tokens if frequent pair exists
                update = False
                k = 0
                while k < len(w_tokens) - 1:
                    if (w_tokens[k], w_tokens[k + 1]) == frequent_pair:
                        new_w_tokens.append(k_vocab)
                        k += 2
                        update = True
                    else:
                        new_w_tokens.append(w_tokens[k])
                        k += 1
                if k < len(w_tokens):
                    new_w_tokens.append(w_tokens[-1])
                # Update the pair_counter if word tokens was updated
                if update:
                    # Remove previous counts
                    for a, b in zip(w_tokens, w_tokens[1:]):
                        pair_counter[(a, b)] -= count

                    # Add new counts
                    for a, b in zip(new_w_tokens, new_w_tokens[1:]):
                        if (a, b) in pair_counter.keys():
                            pair_counter[(a, b)] += count
                        else:
                            pair_counter[(a, b)] = count
            pre_tokens[i][0] = new_w_tokens

        # Update pair_counter and check if merge is possible
        pair_counter.pop(frequent_pair)
        if not pair_counter:
            has_merge = False

        pbar.update(1)
    pbar.close()

    return vocab, merges


def train_bpe_tokenizer_simple(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    n_chunks: int = 8,
    n_workers: int = 4,
    parallelizing: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. Compute the pre-tokens
    pre_tokens_counter = Counter()

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, n_chunks, special_tokens)

    if parallelizing:
        # Compute the set of pre-tokens with their counts
        n_workers = min(len(boundaries) - 1, n_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            with open(input_path, "rb") as f:
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    futures.append(executor.submit(_pretokenizer_dict, chunk, special_tokens))

            for _, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                pre_tokens_counter.update(result)
    else:
        # Compute the set of pre-tokens with their counts
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pre_tokens_counter.update(_pretokenizer_dict(chunk, special_tokens))

    # 2. Compute the merges
    pre_tokens: tuple[list[int], int] = [[_word_to_bytes(k), v] for k, v in pre_tokens_counter.items()]

    vocab, merges = _merge_tokens(pre_tokens, special_tokens, vocab_size)

    return vocab, merges


def _compute_pre_tokens(
    input_path: str | os.PathLike,
    special_tokens: list[str] | None,
    n_chunks: int,
    n_workers: int,
    parallelizing: bool,
):
    # 1. Compute the pre-tokens
    pre_tokens_counter = Counter()

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, n_chunks, special_tokens)

    if parallelizing:
        # Compute the set of pre-tokens with their counts
        n_workers = min(len(boundaries) - 1, n_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            with open(input_path, "rb") as f:
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    futures.append(executor.submit(_pretokenizer_dict, chunk, special_tokens))

            for _, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                pre_tokens_counter.update(result)
    else:
        # Compute the set of pre-tokens with their counts
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pre_tokens_counter.update(_pretokenizer_dict(chunk, special_tokens))

    pre_tokens: tuple[list[int], int] = [[_word_to_bytes(k), v] for k, v in pre_tokens_counter.items()]

    return pre_tokens


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = None,
    n_chunks: int = 8,
    n_workers: int = 4,
    parallelizing: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. Compute the pre-tokens
    pre_tokens = _compute_pre_tokens(
        input_path, special_tokens, n_chunks=n_chunks, n_workers=n_workers, parallelizing=parallelizing
    )

    # 2. Compute the merges
    vocab, merges = _merge_tokens_optimized(pre_tokens, special_tokens, vocab_size)

    return vocab, merges
