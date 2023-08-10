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
    subseqs = miditok.pytorch_data.split_seq_in_subsequences(
        seq, min_seq_len, max_seq_len
    )

    assert [i for subseq in subseqs for i in subseq] == seq[
        :300
    ], "Sequence split failed"


def test_dataset_ram():
    multitrack_midis_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))[
        :3
    ]
    one_track_midis_paths = list(Path("tests", "Maestro_MIDIs").glob("**/*.mid"))[:3]
    tokens_os_dir = Path("tests", "multitrack_tokens_os")

    # MIDI + One token stream
    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer_os = miditok.TSD(config)
    dataset_os = miditok.pytorch_data.DatasetTok(
        one_track_midis_paths,
        50,
        100,
        tokenizer_os,
    )
    for _ in dataset_os:
        pass

    # MIDI + Multiple token streams
    tokenizer_ms = miditok.TSD(miditok.TokenizerConfig())
    dataset_ms = miditok.pytorch_data.DatasetTok(
        multitrack_midis_paths,
        50,
        100,
        tokenizer_ms,
    )
    _ = dataset_ms.__repr__()
    dataset_ms.reduce_nb_samples(2)
    assert len(dataset_ms) == 2

    # JSON + one token stream
    if not tokens_os_dir.is_dir():
        tokenizer_os.tokenize_midi_dataset(
            multitrack_midis_paths,
            tokens_os_dir,
        )
    _ = miditok.pytorch_data.DatasetTok(
        list(tokens_os_dir.glob("**/*.json")),
        50,
        100,
    )

    assert True


def test_dataset_io():
    multitrack_midis_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))[
        :3
    ]
    tokens_os_dir = Path("tests", "multitrack_tokens_os")

    if not tokens_os_dir.is_dir():
        config = miditok.TokenizerConfig(use_programs=True)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(multitrack_midis_paths, tokens_os_dir)

    dataset = miditok.pytorch_data.DatasetJsonIO(
        list(tokens_os_dir.glob("**/*.json")),
        100,
    )

    dataset.reduce_nb_samples(2)
    assert len(dataset) == 2

    for _ in dataset:
        pass

    assert True


def test_split_dataset_to_subsequences():
    multitrack_midis_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))[
        :3
    ]
    tokens_os_dir = Path("tests", "multitrack_tokens_os")
    tokens_split_dir = Path("tests", "multitrack_tokens_os_split")
    tokens_split_dir_ms = Path("tests", "multitrack_tokens_ms_split")

    # One token stream
    if not tokens_os_dir.is_dir():
        config = miditok.TokenizerConfig(use_programs=True)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(multitrack_midis_paths, tokens_os_dir)
    miditok.pytorch_data.split_dataset_to_subsequences(
        list(tokens_os_dir.glob("**/*.json")),
        tokens_split_dir,
        50,
        100,
        True,
    )

    # Multiple token streams
    if not tokens_split_dir_ms.is_dir():
        config = miditok.TokenizerConfig(use_programs=False)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(multitrack_midis_paths, tokens_split_dir_ms)
    miditok.pytorch_data.split_dataset_to_subsequences(
        list(tokens_split_dir_ms.glob("**/*.json")),
        tokens_split_dir,
        50,
        100,
        False,
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
        {"input_ids": randint(0, 300, (seq_len,))} for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    # seq_len + 1 as we add 2 tokens (BOS & EOS) but shift labels so -1
    assert list(batch_collated["input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths) + 1,
    ]

    # This time with labels already in batch and embed pooling, padding right
    collator.pad_on_left = False
    batch_from_dataloader = [
        {
            "input_ids": randint(0, 300, (seq_len, 5)),
            "labels": randint(0, 300, (seq_len, 5)),
        }
        for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    assert list(batch_collated["input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths) + 1,
        5,
    ]

    assert True


if __name__ == "__main__":
    test_split_seq()
    test_dataset_ram()
    test_dataset_io()
    test_split_dataset_to_subsequences()
    test_collator()
