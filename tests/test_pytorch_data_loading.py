#!/usr/bin/python3 python

"""
Test classes and methods from the pytorch_data module.
"""

from pathlib import Path

import miditok
from torch import randint


def test_split_seq():
    min_seq_len = 50
    max_seq_len = 100
    seq = list(range(320))
    subseqs = miditok.pytorch_data.split_seq_in_subsequences(seq, min_seq_len, max_seq_len)

    assert [i for subseq in subseqs for i in subseq] == seq[:300], "Sequence split failed"


def test_dataset_ram():
    # One token stream
    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer_os = miditok.TSD(config)
    dataset_os = miditok.pytorch_data.DatasetTok(
        list(Path("tests", "Maestro_MIDIs").glob("**/*.mid")),
        50,
        100,
        tokenizer_os,
    )
    for _ in dataset_os:
        pass

    # Multiple token streams
    tokenizer_ms = miditok.TSD(miditok.TokenizerConfig())
    dataset_ms = miditok.pytorch_data.DatasetTok(
        list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid")),
        50,
        100,
        tokenizer_ms,
    )

    assert True


def test_dataset_io():
    midi_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))[:3]
    tokens_dir = Path("tests", "dataset_io_tokens")

    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer = miditok.TSD(config)
    tokenizer.tokenize_midi_dataset(midi_paths, tokens_dir)

    dataset = miditok.pytorch_data.DatasetJsonIO(
        list(tokens_dir.glob("**/*.json")),
        100,
    )

    for _ in dataset:
        pass

    assert True


def test_split_dataset_to_subsequences():
    midi_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))[:3]
    tokens_dir = Path("tests", "dataset_io_tokens")
    tokens_split_dir = Path("tests", "dataset_io_tokens_split")

    if not tokens_dir.is_dir():
        config = miditok.TokenizerConfig(use_programs=True)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(midi_paths, tokens_dir)

    miditok.pytorch_data.split_dataset_to_subsequences(
        list(tokens_dir.glob("**/*.json")),
        tokens_split_dir,
        50,
        100,
        True,
    )

    assert True


def test_collator():
    collator = miditok.pytorch_data.DataCollator(
        0,
        1,
        2,
        pad_on_left=True,
        copy_inputs_as_labels=True,
        shift_labels=True,
    )
    seq_lengths = [120, 100, 80, 200]

    # Just input ids
    batch_from_dataloader = [
        {"input_ids": randint(0, 300, (seq_len, ))} for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    # seq_len + 1 as we add 2 tokens (BOS & EOS) but shift labels so -1
    assert list(batch_collated["input_ids"].size()) == [len(seq_lengths), max(seq_lengths) + 1]

    # This time with labels already in batch and embed pooling, padding right
    collator.pad_on_left = False
    batch_from_dataloader = [
        {"input_ids": randint(0, 300, (seq_len, 5)),
         "labels": randint(0, 300, (seq_len, 5))}
        for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    assert list(batch_collated["input_ids"].size()) == [len(seq_lengths), max(seq_lengths) + 1, 5]

    assert True


if __name__ == "__main__":
    test_split_seq()
    test_dataset_ram()
    test_dataset_io()
    test_split_dataset_to_subsequences()
    test_collator()
