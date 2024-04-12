#!/usr/bin/python3 python

"""Benchmark for fast BPE with Hugging Face tokenizers library."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path, PurePath
from time import time

from prettytable import PrettyTable
from tqdm import tqdm

import miditok

# Special beat res for test, up to 64 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 4): 8, (4, 12): 4}
ADDITIONAL_TOKENS_TEST = {
    "Chord": False,  # false to speed up tests as it takes some time on maestro MIDIs
    "Rest": True,
    "Tempo": True,
    "TimeSignature": True,
    "Program": False,
    "beat_res_rest": {(0, 16): 4},
    "num_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": {4: [4]},
}


def bpe_benchmark(data_path: str | Path | PurePath = "./tests/Maestro") -> None:
    r"""
    Benchmark BPE encoding, batched and un-batched.

    :param data_path: root path to the data to test
    """
    random.seed(777)
    tokenizations = ["Structured", "REMI", "MIDILike", "TSD"]
    batch_sizes = [1, 16, 64, 128]
    vocab_size = 600
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.midi"))

    def create_tokenizer(tokenization_: str) -> miditok.MIDITokenizer:
        add_tokens_ = deepcopy(ADDITIONAL_TOKENS_TEST)
        return getattr(miditok, tokenization_)(
            beat_res=BEAT_RES_TEST, additional_tokens=add_tokens_
        )

    # Tokenize data
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        tokenizer.tokenize_dataset(
            files, Path("tests", "test_results", f"{tokenization}_maestro")
        )

    # Loading tokens
    data = {tokenization: [] for tokenization in tokenizations}
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        token_files = list(
            Path("tests", "test_results", f"{tokenization}_maestro").glob("**/*.json")
        )
        for file in tqdm(token_files, desc=f"Loading tokens for {tokenization}"):
            tokens = tokenizer.load_tokens(file)["ids"][0]
            data[tokenization].append(miditok.TokSequence(ids=tokens))

    # Record times
    t = PrettyTable(
        ["Tokenization", "Fast train"] + [f"Fast enc. bsz={b}" for b in batch_sizes],
        title="BPE benchmark: Fast = with 🤗tokenizers (Rust), "
        "values are in second, bsz is batch size",
    )
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        row = [tokenization, None]
        row += [None for _ in batch_sizes]

        # Loading tokens
        data = []
        token_files = list(
            Path("tests", "test_results", f"{tokenization}_maestro").glob("**/*.json")
        )
        for file in tqdm(token_files, desc=f"Loading tokens for {tokenization}"):
            tokens = tokenizer.load_tokens(file)["ids"][0]
            data.append(miditok.TokSequence(ids=tokens))

        # Fast BPE learning
        t0 = time()
        tokenizer.train(
            vocab_size=vocab_size,
            files_paths=list(
                Path("tests", "test_results", f"{tokenization}_maestro").glob(
                    "**/*.json"
                )
            ),
            files_lim=None,
            start_from_empty_voc=False,
        )
        t1 = time() - t0
        row[1] = f"{t1:.3f}"
        print(f"Fast BPE learning for {tokenization} took {t1:.3f} sec")

        # Fast BPE tokenization
        for b, batch_size in enumerate(batch_sizes):
            tok_times = []
            for i in tqdm(
                range(0, len(data), batch_size),
                desc=f"Encoding fast BPE, batch size={batch_size}",
            ):
                tokens_bpe = deepcopy(data[i : i + batch_size])
                t0 = time()
                tokenizer.apply_bpe(tokens_bpe)
                tok_times.append((time() - t0) / len(tokens_bpe))  # mean per batch
            mean_time = sum(tok_times) / len(tok_times)
            row[2 + b] = f"{mean_time:.3f}"
            print(
                f"Fast BPE encoding for {tokenization} and batch size of {batch_size}"
                f"took {mean_time:.3f} sec"
            )

        t.add_row(row)

    print(t)


if __name__ == "__main__":
    bpe_benchmark()
