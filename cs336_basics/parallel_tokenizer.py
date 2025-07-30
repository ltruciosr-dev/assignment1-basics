#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.bpe_tokenizer import _find_chunk_boundaries
from tests.common import DATA_PATH, TOKEN_PATH


def line_iterator_from_position(file_path, start_pos, end_pos, encoding="utf-8"):
    """Iterator that yields lines from a specific byte range in the file,
    including any line that overlaps with the range [start_pos, end_pos).
    If end_pos is in the middle of a line, only yield up to end_pos for that line.
    """
    with open(file_path, encoding=encoding) as f:
        f.seek(start_pos)
        while True:
            line_start_pos = f.tell()
            if line_start_pos >= end_pos:
                break
            line = f.readline()
            if not line:  # EOF
                break
            line_end_pos = f.tell()
            if line_end_pos > end_pos:
                num_bytes_to_read = end_pos - line_start_pos
                f.seek(line_start_pos)
                partial = f.read(num_bytes_to_read)
                yield partial
                break
            else:
                yield line


def tokenize_chunk(args):
    """Worker function to tokenize a chunk of the file."""
    file_path, start_pos, end_pos, vocab_path, merges_path, special_tokens = args

    # Create tokenizer instance in worker process
    tokenizer = BPETokenizer.from_pkl_files(vocab_path, merges_path, special_tokens)

    # Process the chunk
    tokens = []
    for token in tokenizer.encode_iterable(line_iterator_from_position(file_path, start_pos, end_pos)):
        tokens.append(token)

    # Return both tokens and bytes processed for progress tracking
    bytes_processed = end_pos - start_pos
    return np.array(tokens, dtype="uint16"), bytes_processed


def process_file_with_boundaries(
    input_path, vocab_path, merges_path, special_tokens=None, n_chunks=8, max_workers=None
):
    """Process file using concurrent.futures with proper chunk boundaries."""

    # Use _find_chunk_boundaries for proper chunking
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, n_chunks, special_tokens=[])

    # Prepare arguments for worker processes
    worker_args = []
    for i in range(len(boundaries) - 1):
        start_pos = boundaries[i]
        end_pos = boundaries[i + 1]
        worker_args.append((input_path, start_pos, end_pos, vocab_path, merges_path, special_tokens))

    # Use total bytes for progress bar
    total_bytes = os.path.getsize(input_path)

    actual_chunks = len(worker_args)
    # Limit max_workers to available processors, not number of chunks
    if max_workers is None:
        max_workers = min(actual_chunks, os.cpu_count() or 1)
    else:
        max_workers = min(max_workers, actual_chunks)

    print(f"Processing file with {actual_chunks} chunks using {max_workers} workers...")
    print(f"File size: {total_bytes:,} bytes")
    if len(boundaries) > 10:
        print(f"Chunk boundaries: {boundaries[:5]} ... {boundaries[-5:]}")
    else:
        print(f"Chunk boundaries: {boundaries}")

    # Process chunks in parallel using concurrent.futures
    results = [None] * actual_chunks  # Pre-allocate to maintain order

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(tokenize_chunk, worker_args[i]): i for i in range(actual_chunks)}

        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Processing") as pbar:
            # Process results as they complete
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    chunk_tokens, bytes_processed = future.result()
                    results[chunk_index] = chunk_tokens
                    pbar.update(bytes_processed)
                except Exception as exc:
                    print(f"Chunk {chunk_index} generated an exception: {exc}")
                    raise exc

    # Concatenate all results in order
    print("Concatenating results...")
    final_tokens = np.concatenate(results)

    return final_tokens


def main():
    """Main function to run the tokenization process."""
    # Configuration (adjust paths as needed)
    # input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    # vocab_path = TOKEN_PATH / "TinyStoriesV2-GPT4-train-vocab-10_000.pkl"
    # merges_path = TOKEN_PATH / "TinyStoriesV2-GPT4-train-merges-10_000.pkl"
    input_path = DATA_PATH / "owt_train.txt"
    vocab_path = TOKEN_PATH / "owt_train-vocab-32000_0.2.pkl"
    merges_path = TOKEN_PATH / "owt_train-merges-32000_0.2.pkl"
    special_tokens = ["<|endoftext|>"]

    # tokens_path = TOKEN_PATH / "TinyStoriesV2-GPT4-train-tokens-10_000.npy"
    tokens_path = TOKEN_PATH / "owt_train-tokens-32000_0.2.npy"
    n_chunks = 8096
    max_workers = 12

    # Check if files exist
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    if not Path(vocab_path).exists() or not Path(merges_path).exists():
        print("ERROR: Tokenizer files not found:")
        print(f"  vocab_path: {vocab_path}")
        print(f"  merges_path: {merges_path}")
        return

    try:
        # Process with multiprocessing using proper boundaries
        tokens = process_file_with_boundaries(
            input_path=input_path,
            vocab_path=vocab_path,
            merges_path=merges_path,
            special_tokens=special_tokens,
            n_chunks=n_chunks,
            max_workers=max_workers,
        )

        print(f"Final result: {len(tokens):,} tokens")
        print(f"Token array shape: {tokens.shape}")
        print(f"Token array dtype: {tokens.dtype}")

        np.save(tokens_path, tokens)
        print(f"Results saved to: {tokens_path}")

    except Exception as e:
        print(f"ERROR during tokenization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
