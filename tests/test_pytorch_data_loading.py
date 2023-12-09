#!/usr/bin/python3 python

"""
Test classes and methods from the pytorch_data module.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

from symusic import Score
from torch import randint

import miditok

from .utils import MIDI_PATHS_MULTITRACK, MIDI_PATHS_ONE_TRACK


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


def test_dataset_ram(
    tmp_path: Path,
    midi_paths_one_track: Optional[Sequence[Union[str, Path]]] = None,
    midi_paths_multitrack: Optional[Sequence[Union[str, Path]]] = None,
):
    if midi_paths_one_track is None:
        midi_paths_one_track = MIDI_PATHS_ONE_TRACK[:3]
    if midi_paths_multitrack is None:
        midi_paths_multitrack = MIDI_PATHS_MULTITRACK[:3]
    tokens_os_dir = tmp_path / "multitrack_tokens_os"
    dummy_labels = {
        label: i
        for i, label in enumerate(
            set(path.name.split("_")[0] for path in midi_paths_one_track)
        )
    }

    def get_labels_one_track(_: Sequence, file_path: Path) -> int:
        return dummy_labels[file_path.name.split("_")[0]]

    def get_labels_multitrack(midi: Score, _: Path) -> int:
        return len(midi.tracks)

    def get_labels_multitrack_one_stream(tokens: Sequence, _: Path) -> int:
        return len(tokens) // 4

    # MIDI + One token stream + labels
    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer_os = miditok.TSD(config)
    dataset_os = miditok.pytorch_data.DatasetTok(
        midi_paths_one_track,
        50,
        100,
        tokenizer_os,
        func_to_get_labels=get_labels_one_track,
    )
    # Test iteration, and collator with user labels
    for _ in dataset_os:
        pass
    collator = miditok.pytorch_data.DataCollator(
        0,
        1,
        2,
        pad_on_left=True,
    )
    _ = collator([dataset_os[i] for i in range(4)])

    # MIDI + Multiple token streams + labels
    tokenizer_ms = miditok.TSD(miditok.TokenizerConfig())
    dataset_ms = miditok.pytorch_data.DatasetTok(
        midi_paths_multitrack,
        50,
        100,
        tokenizer_ms,
        func_to_get_labels=get_labels_multitrack,
    )
    _ = dataset_ms.__repr__()
    dataset_ms.reduce_nb_samples(2)
    assert len(dataset_ms) == 2

    # JSON + one token stream
    if not tokens_os_dir.is_dir():
        tokenizer_os.tokenize_midi_dataset(
            midi_paths_multitrack,
            tokens_os_dir,
        )
    _ = miditok.pytorch_data.DatasetTok(
        list(tokens_os_dir.glob("**/*.json")),
        50,
        100,
        func_to_get_labels=get_labels_multitrack_one_stream,
    )


def test_dataset_io(
    tmp_path: Path, midi_path: Optional[Sequence[Union[str, Path]]] = None
):
    if midi_path is None:
        midi_path = MIDI_PATHS_MULTITRACK[:3]
    tokens_os_dir = tmp_path / "multitrack_tokens_os"

    if not tokens_os_dir.is_dir():
        config = miditok.TokenizerConfig(use_programs=True)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(midi_path, tokens_os_dir)

    dataset = miditok.pytorch_data.DatasetJsonIO(
        list(tokens_os_dir.glob("**/*.json")),
        100,
    )

    dataset.reduce_nb_samples(2)
    assert len(dataset) == 2

    for _ in dataset:
        pass


def test_split_dataset_to_subsequences(
    tmp_path: Path,
    midi_path: Optional[Sequence[Union[str, Path]]] = None,
):
    if midi_path is None:
        midi_path = MIDI_PATHS_MULTITRACK[:3]
    tokens_os_dir = tmp_path / "multitrack_tokens_os"
    tokens_split_dir = tmp_path / "multitrack_tokens_os_split"
    tokens_split_dir_ms = tmp_path / "multitrack_tokens_ms_split"

    # One token stream
    if not tokens_os_dir.is_dir():
        config = miditok.TokenizerConfig(use_programs=True)
        tokenizer = miditok.TSD(config)
        tokenizer.tokenize_midi_dataset(midi_path, tokens_os_dir)
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
        tokenizer.tokenize_midi_dataset(midi_path, tokens_split_dir_ms)
    miditok.pytorch_data.split_dataset_to_subsequences(
        list(tokens_split_dir_ms.glob("**/*.json")),
        tokens_split_dir,
        50,
        100,
        False,
    )


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
