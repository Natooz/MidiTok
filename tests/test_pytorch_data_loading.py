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
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from symusic import Score

    from miditok import TokSequence


def get_labels_seq_len(midi: Score, tokseq: TokSequence, _: Path) -> int:
    return len(tokseq) // len(midi.tracks)


def get_labels_seq(midi: Score, tokseq: TokSequence, _: Path) -> list[int]:
    if isinstance(tokseq, list):
        return tokseq[0].ids[: -len(midi.tracks)]
    if len(tokseq) > len(midi.tracks):
        return tokseq.ids[: -len(midi.tracks)]
    return tokseq.ids


@pytest.mark.parametrize(
    "tokenizer_cls", [miditok.TSD, miditok.Octuple], ids=["TSD", "Octuple"]
)
@pytest.mark.parametrize("one_token_stream", [True, False], ids=["1 strm", "n strm"])
@pytest.mark.parametrize("split_midis", [True, False], ids=["split", "no split"])
@pytest.mark.parametrize("pre_tokenize", [True, False], ids=["pretok", "no pretok"])
@pytest.mark.parametrize("func_labels", [get_labels_seq_len, get_labels_seq])
@pytest.mark.parametrize("num_overlap_bars", [0, 1], ids=["no overlap", "overlap"])
def test_dataset_midi(
    tmp_path: Path,
    tokenizer_cls: Callable,
    one_token_stream: bool,
    split_midis: bool,
    pre_tokenize: bool,
    func_labels: Callable,
    num_overlap_bars: int,
    midi_paths: Sequence[Path] = MIDI_PATHS_MULTITRACK,
    max_seq_len: int = 1000,
):
    config = miditok.TokenizerConfig(use_programs=one_token_stream)
    tokenizer = tokenizer_cls(config)

    # Split MIDIs if requested
    # We perform it twice as the second time, the method would return the same paths as
    # the ones created in the first call.
    if split_midis:
        t0 = time()
        midi_paths_split1 = miditok.pytorch_data.split_midis_for_training(
            midi_paths,
            tokenizer,
            tmp_path,
            max_seq_len,
            num_overlap_bars=num_overlap_bars,
        )
        t1 = time() - t0
        print(f"First MIDI split call: {t1:.2f} sec")
        t0 = time()
        midi_paths_split2 = miditok.pytorch_data.split_midis_for_training(
            midi_paths,
            tokenizer,
            tmp_path,
            max_seq_len,
            num_overlap_bars=num_overlap_bars,
        )
        t1 = time() - t0
        print(f"Second MIDI split call: {t1:.2f} sec")

        midi_paths_split1.sort()
        midi_paths_split2.sort()
        assert midi_paths_split1 == midi_paths_split2
        midi_paths = midi_paths_split1

    # Creating the Dataset, splitting MIDIs
    t0 = time()
    dataset = miditok.pytorch_data.DatasetMIDI(
        midi_paths,
        tokenizer,
        max_seq_len,
        tokenizer["BOS_None"],
        tokenizer["EOS_None"],
        pre_tokenize=pre_tokenize,
        func_to_get_labels=func_labels,
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


def test_dataset_json(tmp_path: Path, midi_path: Sequence[Path] | None = None):
    if midi_path is None:
        midi_path = MIDI_PATHS_MULTITRACK[:5]
    tokens_dir_path = tmp_path / "multitrack_tokens_dataset_json"

    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer = miditok.TSD(config)
    if not tokens_dir_path.is_dir():
        tokenizer.tokenize_midi_dataset(midi_path, tokens_dir_path)

    tokens_split_dir_path = tmp_path / "multitrack_tokens_dataset_json_split"
    miditok.pytorch_data.split_dataset_to_subsequences(
        list(tokens_dir_path.glob("**/*.json")),
        tokens_split_dir_path,
        300,
        1000,
    )
    dataset = miditok.pytorch_data.DatasetJSON(
        list(tokens_split_dir_path.glob("**/*.json")),
        1000,
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
    # seq_len - 1 as we shift labels
    assert list(batch_collated["input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths) - 1,
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
        max(seq_lengths) - 1,
        5,
    ]
