"""Utils methods for MIDI/tokens split."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from symusic import Score
from torch import LongTensor
from tqdm import tqdm

from miditok.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE
from miditok.utils import (
    get_bars_ticks,
    get_beats_ticks,
    get_num_notes_per_bar,
    split_midi_per_ticks,
    split_midi_per_tracks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miditok import MIDITokenizer


def split_midis_for_training(
    files_paths: Sequence[Path],
    tokenizer: MIDITokenizer,
    save_dir: Path,
    max_seq_len: int,
    average_num_tokens_per_note: float | None = None,
    minimize_padding: bool = False,
    min_seq_len: int | None = None,
) -> list[Path]:
    """
    Split a list of MIDIs into smaller chunks to use for training.

    MIDI splitting allows to split each MIDI from a dataset into chunks of lengths
    calculated in function of the note densities of its bars in order to reduce the
    padding of the batches, using the
    :py:func:`miditok.pytorch_data.split_midi_per_note_density` method.
    The MIDIs are only split at bars, in order have chunks starting at relevant times.

    MIDI splitting can be performed on a dataset once. This method will save a hidden
    file, with a name corresponding to the hash of the list of file paths, in the
    ``save_dir`` directory. When called, it will first check that this file does not
    already exist, and if it is the case will return the paths to all the MIDI files
    within ``save_dir``.

    **If your tokenizer does not tokenize all tracks in one sequence of tokens**
    (``tokenizer.one_token_stream``), the MIDI tracks will be split independently.

    :param files_paths: paths to MIDI files to split.
    :param tokenizer: tokenizer.
    :param save_dir: path to the directory to save the MIDI splits.
    :param max_seq_len: maximum token sequence length that the model will be trained
        with.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer. If given ``None``, this value will automatically be calculated
        from the first 200 MIDI files with the
        :py:func:`miditok.pytorch_data.get_average_num_tokens_per_note` method.
    :param minimize_padding: will split the MIDIs into chunks in order to minimize
        padding at best, at the cost of parts of bars that will be omitted.
        (default: ``False``)
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the MIDI. (default: ``None``, see default value of
        :py:func:`miditok.pytorch_data.split_midi_per_note_density`)
    :return: the paths to the MIDI splits.
    """
    # Safety checks
    midi_split_hidden_file_path = save_dir / f".{hash(tuple(files_paths))}"
    if midi_split_hidden_file_path.is_file():
        warn(
            f"These MIDI have already been split in the saving directory ({save_dir})."
            f" Skipping MIDI splitting.",
            stacklevel=2,
        )
        return list(save_dir.glob("**/*.mid"))
    if not average_num_tokens_per_note:
        average_num_tokens_per_note = get_average_num_tokens_per_note(
            tokenizer, files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
        )

    # Determine the deepest common subdirectory to replicate file tree
    all_parts = [path.parent.parts for path in files_paths]
    max_depth = max(len(parts) for parts in all_parts)
    root_parts = []
    for depth in range(max_depth):
        if len({parts[depth] for parts in all_parts}) > 1:
            break
        root_parts.append(all_parts[0][depth])
    root_dir = Path(*root_parts)

    # Splitting MIDIs
    new_files_paths = []
    for file_path in tqdm(
        files_paths,
        desc=f"Splitting MIDIs ({save_dir})",
        miniters=int(len(files_paths) / 20),
        maxinterval=480,
    ):
        midis = [Score(file_path)]

        # Separate track first if needed
        tracks_separated = False
        if not tokenizer.one_token_stream and len(midis[0].tracks) > 1:
            midis = split_midi_per_tracks(midis[0])
            tracks_separated = True

        # Split per note density
        for ti, midi_to_split in enumerate(midis):
            midi_splits = split_midi_per_note_density(
                midi_to_split,
                max_seq_len,
                average_num_tokens_per_note,
                minimize_padding,
                min_seq_len,
            )

            # Save them
            for _i, midi_to_save in enumerate(midi_splits):
                # Skip it if there are no notes, this can happen with
                # portions of tracks with no notes but tempo/signature
                # changes happening later
                if len(midi_to_save.tracks) == 0 or midi_to_save.note_num() == 0:
                    continue
                if tracks_separated:
                    file_name = f"{file_path.stem}_t{ti}_{_i}.mid"
                else:
                    file_name = f"{file_path.stem}_{_i}.mid"
                # use with_stem when dropping support for python 3.8
                saving_path = (
                    save_dir / file_path.relative_to(root_dir).parent / file_name
                )
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                midi_to_save.dump_midi(saving_path)
                new_files_paths.append(saving_path)

    # Save file in save_dir to indicate MIDI split has been performed
    with midi_split_hidden_file_path.open("w") as f:
        f.write(f"{len(files_paths)} files after MIDI splits")

    return new_files_paths


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
    minimize_padding: bool = False,
    min_seq_len: int | None = None,
) -> list[Score]:
    """
    Split a MIDI (at bars) into chunks depending on their note densities.

    This method aims to split MIDIs at bars to reduce the amount of padding to apply to
    batches during training. It offers several parameters to control where to split
    depending on the desired outcome, e.g. reduce padding or keep the largest amount of
    data at the cost of padding.

    This method will estimate the number of tokens for each bar depending on the
    tokenizer's average number of tokens per note (tpn), will loop over the
    estimated number of tokens per bar to determine the bars at which the MIDI
    will be "cut".

    When the ongoing MIDI chunk has an estimated token sequence length exceeding
    ``max_seq_len``, the cut will be made at the end of the current bar if:
    * ``minimize_padding``;
    * ``num_tokens_next - max_seq_len > tpb / 2``, i.e. the token sequence would cover
    at least half of the current bar;
    * the current bar is the last one and the last token sequence length would make at
    least alf the maximum sequence length, this to not create orphan MIDIs of very
    short length that would not help to train a model.

    Otherwise, the cut will be at the end of the current bar, thus loosing a few notes.

    :param midi: MIDI to split.
    :param max_seq_len: maximum number of tokens per sequence.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer.
    :param minimize_padding: will split the MIDIs into chunks in order to minimize
        padding at best, at the cost of parts of bars that will be omitted.
        (default: ``False``)
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the MIDI. (default: ``max_seq_len // 2``)
    :return: the list of split MIDIs.
    """
    if min_seq_len is None:
        min_seq_len = max_seq_len // 2
    bar_ticks = get_bars_ticks(midi)
    num_notes_per_bar = get_num_notes_per_bar(midi)
    num_tokens_per_bar = [
        npb * average_num_tokens_per_note for npb in num_notes_per_bar
    ]

    ticks_split = []
    num_tokens_current_chunk = 0
    for bi, tpb in enumerate(num_tokens_per_bar):
        num_tokens_next = num_tokens_current_chunk + tpb
        if num_tokens_next > max_seq_len:
            diff_num_tokens = num_tokens_next - max_seq_len

            # Cut at the **beginning** of the current bar
            if not minimize_padding and (
                diff_num_tokens < tpb / 2  # would cover short part of the bar
                or (bi + 1 == len(bar_ticks) and diff_num_tokens < min_seq_len)
            ):
                ticks_split.append(bar_ticks[bi])
                num_tokens_current_chunk = tpb
            # Cut at the **end** of the current bar
            elif bi + 1 < len(bar_ticks):
                ticks_split.append(bar_ticks[bi + 1])
                num_tokens_current_chunk = 0
        else:
            num_tokens_current_chunk = num_tokens_next

    if len(ticks_split) == 0:
        return [midi]
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
