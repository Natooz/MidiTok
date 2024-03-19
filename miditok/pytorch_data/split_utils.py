"""Utils methods for MIDI/tokens split."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from symusic import Score
from torch import LongTensor
from tqdm import tqdm

from miditok.utils import (
    get_bars_ticks,
    get_beats_ticks,
    get_num_notes_per_bar,
    split_midi_per_ticks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miditok import MIDITokenizer


def get_distribution_num_tokens_per_beat(
    files_paths: Sequence[Path], tokenizer: MIDITokenizer
) -> list[float]:
    """
    Return the distributions of number of tokens per beat for a list of files.

    :param files_paths: paths to the files to load.
    :param tokenizer: tokenizer.
    :return: the distribution of number of tokens per beat for each file, and/or
        each track if ``tokenizer.one_token_stream`` is ``False``.
    """
    tpb_dist = []
    for files_path in tqdm(
        files_paths, desc="Calculating the tokens/beat distribution"
    ):
        # Load MIDI and tokenize it
        midi = Score(files_path)
        ticks_beats = get_beats_ticks(midi)
        num_beats_global = len(ticks_beats)
        tokens = tokenizer(midi)

        if tokenizer.one_token_stream:
            tpb_dist.append(len(tokens) / num_beats_global)
        else:
            ticks_beats = np.array(get_beats_ticks(midi))
            for track, seq in zip(midi.tracks, tokens):
                # track.start is always 0, so we use the first note's time
                beat_start = np.where((ticks_beats - track.notes[0].time) <= 0)[0][-1]
                beat_end = np.where((ticks_beats - track.end()) >= 0)[0]
                beat_end = num_beats_global if len(beat_end) == 0 else beat_end[0]
                tpb_dist.append(len(seq) / (beat_end - beat_start))

    return tpb_dist


def get_num_beats_for_token_seq_len(
    files_paths: Sequence[Path],
    tokenizer: MIDITokenizer,
    sequence_length: int,
    ratio_of_full_sequence_length: float,
) -> float:
    """
    Return the number of beats covering *x*% of the sequences of *y* tokens.

    This method calls
    :py:func:`miditok.pytorch_data.get_num_tokens_per_beat_distribution` and returns
    the number of beats covering ``ratio_of_data_to_keep``% of the sequences of
    ``sequence_length`` tokens.
    This method is useful to calculate the appropriate chunk length in beats to use
    with the :py:func:`miditok.utils.split_midi` method (also called by
    :class:`miditok.pytorch_data.DatasetTok`).

    :param files_paths: paths to the files to load.
    :param tokenizer: tokenizer.
    :param sequence_length: number of tokens in the sequence.
    :param ratio_of_full_sequence_length: ratio of sequences that must contain at most
        `sequence_length`` tokens.
    :return: number of beats covering *x*% of the sequences of *y* tokens.
    """
    tpb_dist = get_distribution_num_tokens_per_beat(files_paths, tokenizer)
    bpt_dist = np.reciprocal(np.array(tpb_dist)) * sequence_length
    bpt_dist.sort()
    return np.percentile(bpt_dist, ratio_of_full_sequence_length * 100)


def split_midi_per_note_density(
    midi: Score,
    max_seq_len: int,
    average_num_tokens_per_note: float,
) -> list[Score]:
    """
    Split a MIDI into chunks depending on their note density.

    Using note densities aims to reduce the amount of padding when splitting the MIDIs.
    # TODO splitting: minimize padding or maximize amount of data --> sort samples?

    :param midi: MIDI to split.
    :param max_seq_len: maximum number of tokens per sequence.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer.
    :return: the list of split MIDIs.
    """
    bar_ticks = get_bars_ticks(midi)
    num_notes_per_bar = get_num_notes_per_bar(midi)
    num_tokens_per_bar = [
        npb * average_num_tokens_per_note for npb in num_notes_per_bar
    ]
    ticks_split = []
    num_tokens = 0
    for bi, tpb in enumerate(num_tokens_per_bar):
        next_num_tokens = num_tokens + tpb
        if next_num_tokens > max_seq_len:
            ticks_split.append(bar_ticks[bi])  # TODO bi good??
            num_tokens = 0
        else:
            num_tokens = next_num_tokens

    return split_midi_per_ticks(midi, ticks_split)


def get_average_num_tokens_per_note(
    tokenizer: MIDITokenizer, files_paths: Sequence[Path]
) -> float:
    """
    Return the average number of tokens per note (tpn) for a list of MIDIs.

    With BPE, the average tpn is likely to be very low.

    :param tokenizer: tokenizer.
    :param files_paths: list of MIDI file paths.
    :return: the average tokens per note.
    """
    num_tokens_per_note = []
    for file_path in files_paths:
        midi = Score(file_path)
        tok_seq = tokenizer(midi)
        if tokenizer.one_token_stream:
            num_notes = midi.note_num()
            num_tokens_per_note.append(len(tok_seq) / num_notes)
        else:
            for track, seq in zip(midi.tracks, tok_seq):
                # TODO test this
                num_tokens_per_note.append(len(seq) / track.note_num())

    return sum(num_tokens_per_note) / len(num_tokens_per_note)


def split_seq_in_subsequences(
    seq: Sequence[any], min_seq_len: int, max_seq_len: int
) -> list[Sequence[Any]]:
    r"""
    Split a sequence of tokens into subsequences.

    The subsequences will have lengths comprised between ``min_seq_len`` and
    ``max_seq_len``: ``min_seq_len <= len(sub_seq) <= max_seq_len``.

    :param seq: sequence to split.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :return: list of subsequences.
    """
    sub_seq = []
    i = 0
    while i < len(seq):
        if i >= len(seq) - min_seq_len:
            break  # last sample is too short
        sub_seq.append(LongTensor(seq[i : i + max_seq_len]))
        i += len(sub_seq[-1])  # could be replaced with max_seq_len

    return sub_seq


def split_dataset_to_subsequences(
    files_paths: Sequence[Path | str],
    out_dir: Path | str,
    min_seq_len: int,
    max_seq_len: int,
    one_token_stream: bool = True,
) -> None:
    """
    Split a dataset of tokens files into subsequences.

    This method is particularly useful if you plan to use a
    :class:`miditok.pytorch_data.DatasetJsonIO`, as it would split token sequences
    into subsequences with the desired lengths before loading them for training.

    :param files_paths: list of files of tokens to split.
    :param out_dir: output directory to save the subsequences.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :param one_token_stream: give False if the token files contains multiple tracks,
        i.e. the first dimension of the value of the "ids" entry corresponds to several
        tracks. Otherwise, leave False. (default: True)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files_paths:
        with Path(file_path).open() as json_file:
            tokens = json.load(json_file)

        # Split sequence(s)
        if one_token_stream:
            subseqs = split_seq_in_subsequences(tokens["ids"], min_seq_len, max_seq_len)
        else:
            subseqs = []
            for track_seq in tokens["ids"]:
                subseqs += split_seq_in_subsequences(
                    track_seq, min_seq_len, max_seq_len
                )

        # Save subsequences
        for i, subseq in enumerate(subseqs):
            path = out_dir / f"{file_path.name}_{i}.json"
            with path.open("w") as outfile:
                new_tok = deepcopy(tokens)
                new_tok["ids"] = subseq
                json.dump(tokens, outfile)
