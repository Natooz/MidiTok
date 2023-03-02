#!/usr/bin/python3 python

"""Benchmark comparing slow VS fast BPE.
"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
from time import time
import random

import miditok
from tqdm import tqdm
from prettytable import PrettyTable

# Special beat res for test, up to 64 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 4): 8, (4, 12): 4}
ADDITIONAL_TOKENS_TEST = {
    "Chord": False,  # set false to speed up tests as it takes some time on maestro MIDIs
    "Rest": True,
    "Tempo": True,
    "TimeSignature": True,
    "Program": False,
    "rest_range": (4, 16),
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": (16, 2),
}


def bpe_benchmark(
    data_path: Union[str, Path, PurePath] = "./tests/Maestro"
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    :param data_path: root path to the data to test
    """
    random.seed(777)
    tokenizations = ["Structured", "REMI", "MIDILike", "TSD"]
    batch_sizes = [1, 16, 64, 128]
    vocab_size = 1000
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.midi"))

    def create_tokenizer(tokenization_: str) -> miditok.MIDITokenizer:
        add_tokens_ = deepcopy(ADDITIONAL_TOKENS_TEST)
        return getattr(miditok, tokenization_)(beat_res=BEAT_RES_TEST, additional_tokens=add_tokens_)

    # Tokenize data
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        tokenizer.tokenize_midi_dataset(files, Path("tests", "test_results", f"{tokenization}_maestro"))

    # Loading tokens
    data = {tokenization: [] for tokenization in tokenizations}
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        token_files = list(Path("tests", "test_results", f"{tokenization}_maestro").glob("**/*.json"))
        for file in tqdm(token_files, desc=f"Loading tokens for {tokenization}"):
            tokens = tokenizer.load_tokens(file)["ids"][0]
            data[tokenization].append(miditok.TokSequence(ids=tokens))

    # Record times
    t = PrettyTable(["Tokenization", "Slow train", "Fast train", "Slow Encoding"] +
                    [f"Fast encoding (bsz {b}" for b in batch_sizes])
    for tokenization in tokenizations:
        tokenizer = create_tokenizer(tokenization)
        row = [tokenization, None, None, None]
        row += [None for _ in batch_sizes]

        # Loading tokens
        data = []
        token_files = list(Path("tests", "test_results", f"{tokenization}_maestro").glob("**/*.json"))
        for file in tqdm(token_files, desc=f"Loading tokens for {tokenization}"):
            tokens = tokenizer.load_tokens(file)["ids"][0]
            data.append(miditok.TokSequence(ids=tokens))

        # Fast BPE learning
        t0 = time()
        tokenizer.learn_bpe(
            vocab_size=vocab_size,
            tokens_paths=list(Path("tests", "test_results", f"{tokenization}_maestro").glob("**/*.json")),
            files_lim=None,
            load_all_token_files_once=True,
            start_from_empty_voc=False,
        )
        t1 = time() - t0
        row[2] = t1
        print(f"Fast BPE learning for {tokenization} took {t1:.2f} sec")

        # Fast BPE tokenization
        for b, batch_size in enumerate(batch_sizes):
            tok_times = []
            for i in tqdm(range(0, len(data), batch_size),
                          desc=f"Encoding fast BPE, batch size={batch_size}"):
                tokens_bpe = deepcopy(data[i: i + batch_size])
                t0 = time()
                tokenizer.apply_bpe(tokens_bpe)
                tok_times.append((time() - t0) / len(tokens_bpe))  # mean per batch
            mean_time = sum(tok_times) / len(tok_times)
            row[4 + b] = mean_time
            print(f"Fast BPE encoding for {tokenization} and batch size of {batch_size} took {mean_time:.2f} sec")

        # Slow BPE learning
        tokenizer = create_tokenizer(tokenization)
        t0 = time()
        tokenizer.learn_bpe_slow(
            Path("tests", "test_results", f"{tokenization}_maestro"),
            vocab_size=vocab_size
        )
        t1 = time() - t0
        row[1] = t1
        print(f"Slow BPE learning for {tokenization} took {t1:.2f} sec")

        # Slow BPE encoding
        tok_time = 0
        for tokens in tqdm(data, desc="Encoding slow BPE"):
            tokens_bpe = deepcopy(tokens)
            t0 = time()
            tokenizer.apply_bpe(tokens_bpe)
            tok_time += time() - t0
        mean_time = tok_time / len(data)
        row[3] = mean_time
        print(f"Slow BPE encoding for {tokenization} took {mean_time:.2f} sec")

        t.add_row(row)

    print(t)


if __name__ == "__main__":
    bpe_benchmark()
