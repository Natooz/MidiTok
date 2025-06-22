"""Test classes and methods from the pytorch_data module."""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import pytest
from torch import randint
from torch.utils.data import DataLoader

import miditok

from .utils_tests import (
    ABC_PATHS,
    MAX_BAR_EMBEDDING,
    MIDI_PATHS_CORRUPTED,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from symusic import Score


def get_labels_seq_len(score: Score, tokseq: miditok.TokSequence, _: Path) -> int:
    num_track = 1 if len(score.tracks) == 0 else len(score.tracks)
    if isinstance(tokseq, miditok.TokSequence):
        return len(tokseq) // num_track
    return len(tokseq[0]) // num_track


def get_labels_seq(score: Score, tokseq: miditok.TokSequence, _: Path) -> list[int]:
    if isinstance(tokseq, list):
        return tokseq[0].ids[: -len(score.tracks)]
    if len(tokseq) > len(score.tracks):
        return tokseq.ids[: -len(score.tracks)]
    return tokseq.ids


@pytest.mark.parametrize(
    "tokenizer_cls", [miditok.TSD, miditok.Octuple], ids=["TSD", "Octuple"]
)
@pytest.mark.parametrize(
    "one_token_stream_for_programs", [True, False], ids=["1 strm", "n strm"]
)
@pytest.mark.parametrize("split_files", [True, False], ids=["split", "no split"])
@pytest.mark.parametrize("pre_tokenize", [True, False], ids=["pretok", "no pretok"])
@pytest.mark.parametrize("ac_random_tracks_ratio", [None, (0.0, 1.0)])
@pytest.mark.parametrize("ac_random_bars_ratio", [None, (0.0, 1.0)])
@pytest.mark.parametrize("func_labels", [get_labels_seq_len, get_labels_seq])
@pytest.mark.parametrize("num_overlap_bars", [0, 1], ids=["no overlap", "overlap"])
def test_dataset_midi(
    tmp_path: Path,
    tokenizer_cls: Callable,
    one_token_stream_for_programs: bool,
    split_files: bool,
    pre_tokenize: bool,
    ac_random_tracks_ratio: tuple[float, float] | None,
    ac_random_bars_ratio: tuple[float, float] | None,
    func_labels: Callable,
    num_overlap_bars: int,
):
    max_seq_len = 1000
    files_paths = (
        MIDI_PATHS_MULTITRACK + MIDI_PATHS_ONE_TRACK + MIDI_PATHS_CORRUPTED + ABC_PATHS
    )
    config = miditok.TokenizerConfig(
        use_programs=True,
        one_token_stream_for_programs=one_token_stream_for_programs,
        max_bar_embedding=MAX_BAR_EMBEDDING,
    )
    tokenizer = tokenizer_cls(config)

    # Split files if requested
    # We perform it twice as the second time, the method would return the same paths as
    # the ones created in the first call.
    if split_files:
        t0 = time()
        file_paths_split1 = miditok.utils.split_files_for_training(
            files_paths,
            tokenizer,
            tmp_path,
            max_seq_len,
            num_overlap_bars=num_overlap_bars,
        )
        t1 = time() - t0
        print(f"First Score split call: {t1:.2f} sec")
        t0 = time()
        file_paths_split2 = miditok.utils.split_files_for_training(
            files_paths,
            tokenizer,
            tmp_path,
            max_seq_len,
            num_overlap_bars=num_overlap_bars,
        )
        t1 = time() - t0
        print(f"Second Score split call: {t1:.2f} sec")

        file_paths_split1.sort()
        file_paths_split2.sort()
        assert file_paths_split1 == file_paths_split2
        files_paths = file_paths_split1

    # Creating the Dataset, splitting MIDIs
    t0 = time()
    dataset = miditok.pytorch_data.DatasetMIDI(
        files_paths,
        tokenizer,
        max_seq_len,
        tokenizer["BOS_None"],
        tokenizer["EOS_None"],
        pre_tokenize=pre_tokenize,
        ac_tracks_random_ratio_range=ac_random_tracks_ratio,
        ac_bars_random_ratio_range=ac_random_bars_ratio,
        func_to_get_labels=func_labels,
    )
    t1 = time() - t0
    print(f"Dataset init took {t1:.2f} sec")

    # Test iteration, and collator with user labels
    batch = [dataset[i] for i in range(min(len(dataset), 10))]

    # Test with DataLoader and collator
    collator = miditok.pytorch_data.DataCollator(
        tokenizer.pad_token_id,
        pad_on_left=True,
    )
    _ = collator(batch)
    dataloader = DataLoader(dataset, 16, collate_fn=collator)
    for _ in dataloader:
        pass


def test_dataset_json(tmp_path: Path):
    file_paths = MIDI_PATHS_MULTITRACK[:5]
    tokens_dir_path = tmp_path / "multitrack_tokens_dataset_json"

    config = miditok.TokenizerConfig(use_programs=True)
    tokenizer = miditok.TSD(config)
    if not tokens_dir_path.is_dir():
        tokenizer.tokenize_dataset(file_paths, tokens_dir_path)

    tokens_split_dir_path = tmp_path / "multitrack_tokens_dataset_json_split"
    miditok.utils.split_tokens_files_to_subsequences(
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

    # Encoder and decoder input ids
    batch_from_dataloader = [
        {
            "input_ids": randint(0, 300, (seq_len,)),
            "decoder_input_ids": randint(0, 300, (seq_len,)),
        }
        for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    # seq_len - 1 as we shift labels
    assert list(batch_collated["input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths),
    ]
    assert list(batch_collated["decoder_input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths) - 1,
    ]

    # This time with labels already in batch and embed pooling, padding right
    collator.pad_on_left = False
    batch_from_dataloader = [
        {
            "input_ids": randint(0, 300, (seq_len, 5)),
            "decoder_input_ids": randint(0, 300, (seq_len, 5)),
            "labels": randint(0, 300, (seq_len, 5)),
        }
        for seq_len in seq_lengths
    ]
    batch_collated = collator(batch_from_dataloader)
    assert list(batch_collated["input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths),
        5,
    ]
    assert list(batch_collated["decoder_input_ids"].size()) == [
        len(seq_lengths),
        max(seq_lengths) - 1,
        5,
    ]
    assert list(batch_collated["labels"].size()) == [
        len(seq_lengths),
        max(seq_lengths) - 1,
        5,
    ]
