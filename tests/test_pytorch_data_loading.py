"""Test classes and methods from the pytorch_data module."""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import pytest
from torch import randint
from torch.utils.data import DataLoader

import miditok

from .utils_tests import MIDI_PATHS_MULTITRACK

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from symusic import Score

    from miditok import TokSequence


def test_get_num_beats_for_token_seq_len(
    file_paths: Sequence[Path] = MIDI_PATHS_MULTITRACK,
    sequence_length: int = 1000,
    ratio_data: float = 0.8,
):
    tokenizer1 = miditok.TSD()
    tokenizer2 = miditok.TSD(miditok.TokenizerConfig(use_programs=True))
    _ = miditok.pytorch_data.get_num_beats_for_token_seq_len(
        file_paths, tokenizer1, sequence_length, ratio_data
    )
    _ = miditok.pytorch_data.get_num_beats_for_token_seq_len(
        file_paths, tokenizer2, sequence_length, ratio_data
    )


def get_labels_seq_len(midi: Score, tokseq: TokSequence, _: Path) -> int:
    return len(tokseq) // len(midi.tracks)


# TODO param labels func int/list/tensor
@pytest.mark.parametrize("one_token_stream", [True, False], ids=["1 strm", "n strms"])
@pytest.mark.parametrize("split_midis", [True, False], ids=["split", "no split"])
@pytest.mark.parametrize("pre_tokenize", [True, False], ids=["pretok", "no pretok"])
def test_dataset_midi(
    tmp_path: Path,
    one_token_stream: bool,
    split_midis: bool,
    pre_tokenize: bool,
    midi_paths: Sequence[Path] = MIDI_PATHS_MULTITRACK,
    max_seq_len: int = 500,
):
    # Creating the Dataset, splitting MIDIs
    config = miditok.TokenizerConfig(use_programs=one_token_stream)
    tokenizer = miditok.TSD(config)
    t0 = time()
    dataset = miditok.pytorch_data.DatasetMIDI(
        midi_paths,
        tokenizer,
        max_seq_len,
        tokenizer["BOS_None"],
        tokenizer["EOS_None"],
        split_midis=split_midis,
        save_dir=tmp_path,
        pre_tokenize=pre_tokenize,
        func_to_get_labels=get_labels_seq_len,
    )
    t1 = time() - t0
    print(f"Dataset init took {t1:.2f} sec")

    # Test iteration, and collator with user labels
    batch = [dataset[i] for i in range(min(len(dataset), 10))]

    # Test with DataLoader and collator
    collator = miditok.pytorch_data.DataCollator(
        tokenizer["PAD_None"],
        pad_on_left=True,
    )
    _ = collator(batch)
    dataloader = DataLoader(dataset, 16, collate_fn=collator)
    for _ in dataloader:
        pass

    # Recreate the dataset, if the MIDI split is already done in tmp_path, it shouldn't
    # be done twice. This second init should be faster than the first.
    if split_midis:
        t0 = time()
        dataset2 = miditok.pytorch_data.DatasetMIDI(
            midi_paths,
            tokenizer,
            max_seq_len,
            tokenizer["BOS_None"],
            tokenizer["EOS_None"],
            split_midis=split_midis,
            save_dir=tmp_path,
            pre_tokenize=pre_tokenize,  # not useful but just to make sure
            func_to_get_labels=get_labels_seq_len,
        )
        t1 = time() - t0
        print(f"Second Dataset init took {t1:.2f} sec")

        dataset.files_paths.sort()
        dataset2.files_paths.sort()
        assert dataset.files_paths == dataset2.files_paths


def test_dataset_jsonio(tmp_path: Path, midi_path: Sequence[Path] | None = None):
    if midi_path is None:
        midi_path = MIDI_PATHS_MULTITRACK[:3]
    tokens_os_dir = tmp_path / "multitrack_tokens_os"

    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer = miditok.TSD(config)
    if not tokens_os_dir.is_dir():
        tokenizer.tokenize_midi_dataset(midi_path, tokens_os_dir)

    dataset = miditok.pytorch_data.DatasetJsonIO(
        list(tokens_os_dir.glob("**/*.json")),
        100,
        tokenizer["BOS_None"],
        tokenizer["EOS_None"],
    )

    for _ in dataset:
        pass


def test_collator():
    collator = miditok.pytorch_data.DataCollator(
        0,
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
