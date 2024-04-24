#!/usr/bin/python3 python

"""Benchmark for training a tokenizer with the Hugging Face tokenizers library."""

from __future__ import annotations

import random
from pathlib import Path
from time import time
from typing import Literal

import numpy as np
from pandas import DataFrame, read_csv  # requires tabulate package
from symusic import Score
from tqdm import tqdm

import miditok
from benchmarks.utils import mean_std_str
from miditok.constants import SCORE_LOADING_EXCEPTION

# Tokenizer
TOKENIZER_PARAMS = {
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "base_tokenizer": "REMI",
}
SPLITS = ["no", "bar", "beat"]
WORDPIECE_MAX_CHARS_PER_WORD = [100, 200, 500, 1000, 2000, 5000]

# Data
DATASETS = ["Maestro", "Lakh"]
MAX_NUM_FILES_PER_DATASET = 2000

# Training
TOKENIZATION = ["REMI", "TSD", "MIDILike"]
MODELS: list[Literal["BPE", "Unigram", "WordPiece"]] = ["BPE", "Unigram", "WordPiece"]
VOCAB_SIZE = 20000
MAX_NUM_FILES_TRAINING = 20000

# Encoding-decoding
BATCH_SIZES = [1, 16, 64, 128]

# Other
SEED = 777
BENCHMARK_PATH = Path("benchmarks", "tokenizer_training")
RESULTS_PATH = BENCHMARK_PATH / "results"


def dataset_files_paths(dataset_name: str) -> list[Path]:
    """
    Return the paths of the files of a dataset.

    :param dataset_name: name of the dataset. Must correspond to the directory name.
    :return: paths of the files of the dataset.
    """
    data_path = Path("..", "data", dataset_name).resolve()
    return list(data_path.glob("**/*.mid")) + list(data_path.glob("**/*.midi"))


def seq_len_splits(datasets_params: list[tuple[str, dict, str]]) -> None:
    """
    Measure the average token sequence lengths after splitting per bars or beats.

    Measures the average token sequence length (in base tokens) after splitting the
    token sequence of whole files into bars or beats.
    These measures can be used to chose good `max_input_chars_per_word` values for
    WordPiece.
    """
    indexes = [f"{tok} - {split}" for tok in TOKENIZATION for split in SPLITS[1:]]
    columns = [data[2] for data in datasets_params]
    df = DataFrame(index=indexes, columns=columns)

    # Perform measures
    for dataset, tok_params, col_name in datasets_params:
        files_paths = dataset_files_paths(dataset)
        for tokenization in TOKENIZATION:
            tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
                tokenizer_config=miditok.TokenizerConfig(**tok_params)
            )

            lengths_subseqs_bars, lengths_subseqs_beats = [], []
            pbar_desc = f"Analyzing {dataset} for {tokenization}"
            count = 0
            for file_path in tqdm(
                files_paths,
                desc=pbar_desc,
                total=min(MAX_NUM_FILES_PER_DATASET, len(files_paths)),
            ):
                try:
                    file = Score(file_path)
                except SCORE_LOADING_EXCEPTION:
                    continue
                tokseqs = tokenizer(file)
                if isinstance(tokseqs, miditok.TokSequence):
                    tokseqs = [tokseqs]
                for tokseq in tokseqs:
                    subseqs_bars = tokseq.split_per_bars()
                    subseqs_beats = tokseq.split_per_beats()
                    lengths_subseqs_bars += [len(subseq) for subseq in subseqs_bars]
                    lengths_subseqs_beats += [len(subseq) for subseq in subseqs_beats]
                count += 1
                if count >= MAX_NUM_FILES_PER_DATASET:
                    break

            avg_std_bars = mean_std_str(lengths_subseqs_bars, 1)
            avg_std_beats = mean_std_str(lengths_subseqs_beats, 1)
            df.at[
                f"{tokenization} - bar", col_name
            ] = f"{avg_std_bars} (↑ {max(lengths_subseqs_bars)})"
            df.at[
                f"{tokenization} - beat", col_name
            ] = f"{avg_std_beats} (↑ {max(lengths_subseqs_beats)})"

    # Save results
    df.to_csv(RESULTS_PATH / "seq_split_bar_beats_lengths.csv")
    df.to_markdown(RESULTS_PATH / "seq_split_bar_beats_lengths.md")
    df.to_latex(RESULTS_PATH / "seq_split_bar_beats_lengths.txt")


def benchmark_training_time() -> None:
    r"""Benchmark BPE encoding, batched and un-batched."""
    indexes = [f"{model} {split}-split" for model in MODELS for split in SPLITS]
    df_file_path = RESULTS_PATH / "training_time.csv"
    if df_file_path.is_file():
        df = read_csv(df_file_path, index_col=0)
    else:
        columns = [f"{dataset} {tok}" for dataset in DATASETS for tok in TOKENIZATION]
        df = DataFrame(index=indexes, columns=columns)

    # Perform measures
    for dataset in DATASETS:
        files_paths = dataset_files_paths(dataset)
        for tokenization in TOKENIZATION:
            col_name = f"{dataset} {tokenization}"
            for model in MODELS:
                for split in SPLITS:
                    index_name = f"{model} {split}-split"

                    # Check measure is not already performed
                    cell_val = df.at[index_name, col_name]
                    if cell_val == cell_val:  # not nan
                        continue

                    # Creates tokenizer
                    tok_params = TOKENIZER_PARAMS.copy()
                    tok_params["encode_ids_split"] = split
                    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
                        tokenizer_config=miditok.TokenizerConfig(**TOKENIZER_PARAMS)
                    )

                    # Trains it
                    random.seed(SEED)
                    t0 = time()
                    tokenizer.train(
                        vocab_size=VOCAB_SIZE,
                        model=model,
                        files_paths=files_paths,
                    )
                    tt = time()

                    # Saves time and tokenizer
                    df.at[index_name, col_name] = f"{tt - t0:.1f} sec"
                    df.to_csv(df_file_path)
                    tokenizer_filename = (
                        f"{tokenization}_{model}_{split}-split_{dataset}.json"
                    )
                    tokenizer.save_params(
                        BENCHMARK_PATH / "tokenizers" / tokenizer_filename
                    )

    # Save results
    df.to_markdown(RESULTS_PATH / "training_time.md")
    df.to_latex(RESULTS_PATH / "training_time.txt")


def benchmark_encoding_decoding_speed_seq_len_reduction() -> None:
    r"""Benchmark encoding-decoding speed and sequence length reduction."""
    df_file_path = RESULTS_PATH / "encoding_time.csv"
    if df_file_path.is_file():
        df_enc_time = read_csv(df_file_path, index_col=0)
        df_dec_time = read_csv(RESULTS_PATH / "decoding_time.csv", index_col=0)
        df_seq_len = read_csv(RESULTS_PATH / "sequence_length.csv", index_col=0)
    else:
        indexes = [f"{model} {split}-split" for model in MODELS for split in SPLITS]
        columns = [
            f"{dataset} {tok} {bs}"
            for dataset in DATASETS
            for tok in TOKENIZATION
            for bs in BATCH_SIZES
        ]
        columns_seq_len = [
            f"{dataset} {tok}" for dataset in DATASETS for tok in TOKENIZATION
        ]
        df_enc_time = DataFrame(index=indexes, columns=columns)
        df_dec_time = DataFrame(index=indexes, columns=columns)
        df_seq_len = DataFrame(index=indexes, columns=columns_seq_len)

    # Perform measures
    for dataset in DATASETS:
        files_paths = dataset_files_paths(dataset)
        for tokenization in TOKENIZATION:
            for model in MODELS:
                for split in SPLITS:
                    index_name = f"{model} {split}-split"
                    col_name = f"{dataset} {tokenization}"

                    # Check measure is not already performed
                    cell_val = df_enc_time.at[index_name, col_name]
                    if cell_val == cell_val:  # not nan
                        continue

                    tokenizer_filename = (
                        f"{tokenization}_{model}_{split}-split_{dataset}.json"
                    )
                    tokenizer = getattr(miditok, tokenization)(
                        params=BENCHMARK_PATH / "tokenizers" / tokenizer_filename
                    )

                    ratios_seq_len = []
                    for batch_size in BATCH_SIZES:
                        col_name_bs = f"{col_name} {batch_size}"
                        times_enc, times_dec = [], []
                        for i in range(0, len(files_paths), batch_size):
                            files_paths_batch = files_paths[i : i + batch_size]
                            tokens = [
                                tokenizer(file_path, encode_ids=False)
                                for file_path in files_paths_batch
                            ]
                            t0 = time()
                            tokenizer.encode_token_ids(tokens)
                            t1 = time()
                            times_enc.append(t1 - t0)

                            if batch_size == BATCH_SIZES[0]:  # only once
                                ratios_seq_len += [
                                    1 - len(seq.ids) / len(seq.tokens) for seq in tokens
                                ]

                            t0 = time()
                            tokenizer.decode_token_ids(tokens)
                            t1 = time()
                            times_dec.append(t1 - t0)

                        df_enc_time.at[index_name, col_name_bs] = mean_std_str(
                            times_enc, 5
                        )
                        df_dec_time.at[index_name, col_name_bs] = mean_std_str(
                            times_dec, 5
                        )
                    df_seq_len.at[index_name, col_name] = mean_std_str(
                        ratios_seq_len, 3
                    )

                    df_enc_time.to_csv(df_file_path)
                    df_dec_time.to_csv(RESULTS_PATH / "decoding_time.csv")
                    df_seq_len.to_csv(RESULTS_PATH / "sequence_length.csv")

    # Save results
    for df, file_name in [
        (df_enc_time, "encoding_time"),
        (df_dec_time, "decoding_time"),
        (df_seq_len, "sequence_length"),
    ]:
        df.to_csv(RESULTS_PATH / f"{file_name}.csv")
        df.to_markdown(RESULTS_PATH / f"{file_name}.md")
        df.to_latex(RESULTS_PATH / f"{file_name}.txt")


def wordpiece_max_chars(datasets_params: list[tuple[str, dict, str]]) -> None:
    """
    Measure the training, encoding and decoding times of WordPiece.

    Measures are made with different `max_input_chars_per_word` values, datasets and
    sequence split to see their impact on training, encoding and decoding times.
    It also measures the ratio of "unknown" tokens resulting of ids encoding,
    measuring the proportion of data covered by these sets of parameters / data.
    """
    indexes = WORDPIECE_MAX_CHARS_PER_WORD
    df_file_path_enc = RESULTS_PATH / "wordpiece_max_chars_enc_time.csv"
    df_file_path_dec = RESULTS_PATH / "wordpiece_max_chars_dec_time.csv"
    df_file_path_train = RESULTS_PATH / "wordpiece_max_chars_train_time.csv"
    if df_file_path_enc.is_file():
        df_enc = read_csv(df_file_path_enc, index_col=0)
        df_dec = read_csv(df_file_path_dec, index_col=0)
        df_train = read_csv(df_file_path_train, index_col=0)
    else:
        columns = [
            f"{data[2]} {split}-split" for data in datasets_params for split in SPLITS
        ]
        df_enc = DataFrame(index=indexes, columns=columns)
        df_dec = DataFrame(index=indexes, columns=columns)
        df_train = DataFrame(index=indexes, columns=columns)
    tokenization = "REMI"

    # Perform measures
    for dataset, tok_params, data_name in datasets_params:
        files_paths = dataset_files_paths(dataset)[:MAX_NUM_FILES_TRAINING]
        for max_chars in WORDPIECE_MAX_CHARS_PER_WORD:
            for split in SPLITS:
                index_name = max_chars
                col_name = f"{data_name} {split}-split"

                # Check measure is not already performed
                cell_val = df_enc.at[index_name, col_name]
                if cell_val == cell_val:  # not nan
                    continue

                # Creates tokenizer
                tok_params = TOKENIZER_PARAMS.copy()
                tok_params["encode_ids_split"] = split
                tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
                    tokenizer_config=miditok.TokenizerConfig(**tok_params)
                )

                # Trains it
                random.seed(SEED)
                train_kwargs = {"max_input_chars_per_word": max_chars}
                t0 = time()
                tokenizer.train(
                    vocab_size=VOCAB_SIZE,
                    model="WordPiece",
                    files_paths=files_paths,
                    **train_kwargs,
                )
                tt = time()
                df_train.at[index_name, col_name] = f"{tt - t0:.1f} sec"

                # Encoding-decoding time
                times_encoding, times_decoding, ratios_unk_tokens = [], [], []
                pbar_desc = (
                    f"Analyzing {dataset} for {max_chars} chars and {split}-split"
                )
                count = 0
                for file_path in tqdm(
                    files_paths,
                    desc=pbar_desc,
                    total=min(MAX_NUM_FILES_PER_DATASET, len(files_paths)),
                ):
                    try:
                        file = Score(file_path)
                    except SCORE_LOADING_EXCEPTION:
                        continue
                    tokseqs = tokenizer.encode(file, encode_ids=False)
                    if isinstance(tokseqs, miditok.TokSequence):
                        tokseqs = [tokseqs]
                    for tokseq in tokseqs:
                        # Encoding
                        t0 = time()
                        tokenizer.encode_token_ids(tokseq)
                        t_enc = time()
                        times_encoding.append(t_enc - t0)
                        # Ratio of unk_token
                        ratio_unk_tokens = len(
                            np.where(np.array(tokseq.ids) == 0)
                        ) / len(tokseq.ids)
                        ratios_unk_tokens.append(ratio_unk_tokens)
                        # Decoding
                        t0 = time()
                        tokenizer.decode_token_ids(tokseq)
                        t_dec = time()
                        times_decoding.append(t_dec - t0)
                    count += 1
                    if count >= MAX_NUM_FILES_PER_DATASET:
                        break

                # Write dataframes
                avg_unk = np.mean(np.array(ratios_unk_tokens))
                df_enc.at[
                    index_name, col_name
                ] = f"{mean_std_str(times_encoding, 4)} ({avg_unk:.3f} unk)"
                df_dec.at[index_name, col_name] = mean_std_str(times_decoding, 4)

                # Saves dataframes
                df_enc.to_csv(df_file_path_enc)
                df_dec.to_csv(df_file_path_dec)
                df_train.to_csv(df_file_path_train)

    df_enc.to_markdown(RESULTS_PATH / "wordpiece_max_chars_enc_time.md")
    df_enc.to_latex(RESULTS_PATH / "wordpiece_max_chars_enc_time.txt")
    df_dec.to_markdown(RESULTS_PATH / "wordpiece_max_chars_dec_time.md")
    df_dec.to_latex(RESULTS_PATH / "wordpiece_max_chars_dec_time.txt")
    df_train.to_markdown(RESULTS_PATH / "wordpiece_max_chars_train_time.md")
    df_train.to_latex(RESULTS_PATH / "wordpiece_max_chars_train_time.txt")


if __name__ == "__main__":
    # Sequence split bar/beat
    tok_params_interleaved = TOKENIZER_PARAMS.copy()
    tok_params_interleaved["use_programs"] = True
    split_data = [
        ("Maestro", TOKENIZER_PARAMS.copy(), "Maestro"),
        ("Lakh", TOKENIZER_PARAMS.copy(), "Lakh monotrack"),
        ("Lakh", tok_params_interleaved.copy(), "Lakh multitrack"),
    ]
    # seq_len_splits(split_data)

    # Training time
    # benchmark_training_time()

    # Encoding-decoding time and sequence length reduction
    # benchmark_encoding_decoding_speed_seq_len_reduction()

    # WordPiece max chars
    wordpiece_data = [
        ("Maestro", TOKENIZER_PARAMS.copy(), "Maestro"),
        ("Lakh", tok_params_interleaved.copy(), "Lakh multitrack"),
    ]
    wordpiece_max_chars(wordpiece_data)
