"""Utils methods for Score/tokens split."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

from symusic import Score, TextMeta
from tqdm import tqdm

from miditok.constants import (
    MAX_NUM_FILES_NUM_TOKENS_PER_NOTE,
    MIDI_FILES_EXTENSIONS,
    SCORE_LOADING_EXCEPTION,
    SUPPORTED_MUSIC_FILE_EXTENSIONS,
)

from .utils import (
    get_bars_ticks,
    get_deepest_common_subdir,
    get_num_notes_per_bar,
    split_score_per_tracks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miditok import MusicTokenizer


def split_files_for_training(
    files_paths: Sequence[Path],
    tokenizer: MusicTokenizer,
    save_dir: Path,
    max_seq_len: int,
    average_num_tokens_per_note: float | None = None,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
) -> list[Path]:
    """
    Split a list of music files into smaller chunks to use for training.

    Splitting files allows to split them into chunks of lengths calculated in function
    of the note densities of its bars in order to reduce the padding of the batches,
    using the :py:func:`miditok.pytorch_data.split_score_per_note_density` method.
    The files are only split at bars, in order have chunks starting at relevant times.

    File splitting can be performed on a dataset once. This method will save a hidden
    file, with a name corresponding to the hash of the list of file paths, in the
    ``save_dir`` directory. When called, it will first check that this file does not
    already exist, and if it is the case will return the paths to all the files within
    ``save_dir``.

    **If your tokenizer does not tokenize all tracks in one sequence of tokens**
    (``tokenizer.one_token_stream``), the music tracks will be split independently.

    :param files_paths: paths to music files to split.
    :param tokenizer: tokenizer.
    :param save_dir: path to the directory to save the files splits.
    :param max_seq_len: maximum token sequence length that the model will be trained
        with.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer. If given ``None``, this value will automatically be calculated
        from the first 200 files with the
        :py:func:`miditok.pytorch_data.get_average_num_tokens_per_note` method.
    :param num_overlap_bars: will create chunks with consecutive overlapping bars. For
        example, if this argument is given ``1``, two consecutive chunks might end at
        the bar *n* and start at the bar *n-1* respectively, thus they will encompass
        the same bar. This allows to create a causality chain between chunks. This value
        should be determined based on the ``average_num_tokens_per_note`` value of the
        tokenizer and the ``max_seq_len`` value, so that it is neither too high nor too
        low. (default: ``1``).
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the file. (default: ``None``, see default value of
        :py:func:`miditok.pytorch_data.split_score_per_note_density`)
    :return: the paths to the files splits.
    """
    # Safety checks
    split_hidden_file_path = save_dir / f".{hash(tuple(files_paths))}"
    if split_hidden_file_path.is_file():
        warn(
            f"These files have already been split in the saving directory ({save_dir})."
            f" Skipping file splitting.",
            stacklevel=2,
        )
        return [
            path
            for path in save_dir.glob("**/*")
            if path.suffix in SUPPORTED_MUSIC_FILE_EXTENSIONS
        ]
    if not average_num_tokens_per_note:
        average_num_tokens_per_note = get_average_num_tokens_per_note(
            tokenizer, files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
        )

    # Determine the deepest common subdirectory to replicate file tree
    root_dir = get_deepest_common_subdir(files_paths)

    # Splitting files
    new_files_paths = []
    for file_path in tqdm(
        files_paths,
        desc=f"Splitting music files ({save_dir})",
        miniters=int(len(files_paths) / 20),
        maxinterval=480,
    ):
        try:
            scores = [Score(file_path)]
        except SCORE_LOADING_EXCEPTION:
            continue

        # Separate track first if needed
        tracks_separated = False
        if not tokenizer.one_token_stream and len(scores[0].tracks) > 1:
            scores = split_score_per_tracks(scores[0])
            tracks_separated = True

        # Split per note density
        for ti, score_to_split in enumerate(scores):
            score_chunks = split_score_per_note_density(
                score_to_split,
                max_seq_len,
                average_num_tokens_per_note,
                num_overlap_bars,
                min_seq_len,
            )

            # Save them
            for _i, chunk_to_save in enumerate(score_chunks):
                # Skip it if there are no notes, this can happen with
                # portions of tracks with no notes but tempo/signature
                # changes happening later
                if len(chunk_to_save.tracks) == 0 or chunk_to_save.note_num() == 0:
                    continue
                # Add a marker to indicate chunk number
                chunk_to_save.markers.append(
                    TextMeta(0, f"miditok: chunk {_i}/{len(score_chunks) - 1}")
                )
                if tracks_separated:
                    file_name = f"{file_path.stem}_t{ti}_{_i}{file_path.suffix}"
                else:
                    file_name = f"{file_path.stem}_{_i}{file_path.suffix}"
                # use with_stem when dropping support for python 3.8
                saving_path = (
                    save_dir / file_path.relative_to(root_dir).parent / file_name
                )
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                if file_path.suffix in MIDI_FILES_EXTENSIONS:
                    chunk_to_save.dump_midi(saving_path)
                else:
                    chunk_to_save.dump_abc(saving_path)
                new_files_paths.append(saving_path)

    # Save file in save_dir to indicate file split has been performed
    with split_hidden_file_path.open("w") as f:
        f.write(f"{len(files_paths)} files after file splits")

    return new_files_paths


def split_score_per_note_density(
    score: Score,
    max_seq_len: int,
    average_num_tokens_per_note: float,
    num_overlap_bars: int = 1,
    min_seq_len: int | None = None,
) -> list[Score]:
    """
    Split a ``symusic.Score`` (at bars) into chunks depending on their note densities.

    This method aims to split music files at bars to reduce the amount of padding to
    apply to batches during training. It offers several parameters to control where to
    split depending on the desired outcome, e.g. reduce padding or keep the largest
    amount of data at the cost of padding.

    This method will estimate the number of tokens for each bar depending on the
    tokenizer's average number of tokens per note (tpn), will loop over the estimated
    number of tokens per bar to determine the bars at which the file will be "cut".

    It is recommended to use this method with a non-zero ``num_overlap_bars``, as
    overlapping allows to keep a form of causality throughout a file from one chunk to
    another. It also reduces padding, but will slightly increase the overall training
    time.

    :param score: ``symusic.Score`` to split.
    :param max_seq_len: maximum number of tokens per sequence.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer.
    :param num_overlap_bars: will create chunks with consecutive overlapping bars. For
        example, if this argument is given ``1``, two consecutive music chunks might
        end at the bar *n* and start at the bar *n-1* respectively, thus they will
        encompass the same bar. This allows to create a causality chain between chunks.
        This value should be determined based on the ``average_num_tokens_per_note``
        value of the tokenizer and the ``max_seq_len`` value, so that it is neither
        too high nor too low. (default: ``1``).
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the file. (default: ``max_seq_len // 4``)
    :return: the list of ``symusic.Score`` chunks.
    """
    if num_overlap_bars < 0:
        msg = (
            f"`num_overlap_bars` must be greater or equal to 0 (received "
            f"{num_overlap_bars})."
        )
        raise ValueError(msg)
    if min_seq_len is None:
        min_seq_len = max_seq_len // 4
    bar_ticks = get_bars_ticks(score)
    num_notes_per_bar = get_num_notes_per_bar(score)
    num_tokens_per_bar = [
        npb * average_num_tokens_per_note for npb in num_notes_per_bar
    ]

    ticks_split = []
    num_tokens_current_chunk = num_bars_current_chunk = 0
    bi = bi_start_chunk = 0
    while bi < len(bar_ticks):
        tpb = num_tokens_per_bar[bi]
        num_tokens_with_current_bar = num_tokens_current_chunk + tpb

        # Cumulative token sequence length exceeds the lim, we need to make a cut
        if num_tokens_with_current_bar > max_seq_len:
            overlap_enabled = True

            # The current bar is the one starting the current chunk and is its token
            # sequence length already exceeds max_seq_len. In this case we cut at the
            # end of the bar and do not apply overlap (otherwise we would end in an
            # infinite loop).
            if bi == bi_start_chunk:
                bi_end_chunk = bi + 1
                overlap_enabled = False

            # Cut at the **beginning** of the current bar:
            # * no overlap and would diff num tokens is low (would lose data);
            elif (
                num_overlap_bars == 0
                and max_seq_len - num_tokens_current_chunk < tpb / 2
            ):
                bi_end_chunk = bi

            # Cut at the **end** of the current bar
            else:
                bi_end_chunk = bi + 1

            # Add tick cut to the list, except if cutting at the end of the last bar
            if bi_end_chunk < len(bar_ticks):
                ticks_split.append((bar_ticks[bi_start_chunk], bar_ticks[bi_end_chunk]))

                # Update values and apply bar overlapping if necessary.
                # We make sure the next bar start is at least one bar after the previous
                bi_start_chunk = max(
                    bi_start_chunk + 1,
                    bi_end_chunk - (num_overlap_bars if overlap_enabled else 0),
                )
                num_tokens_current_chunk = sum(
                    num_tokens_per_bar[i] for i in range(bi_start_chunk, bi + 1)
                )
            else:  # cutting on the last bar --> last chunk is added outside the loop
                num_tokens_current_chunk = num_tokens_with_current_bar

        else:
            num_tokens_current_chunk = num_tokens_with_current_bar
            num_bars_current_chunk += 1

        # Continue to the next bar
        bi += 1

    # Add the last chunk if its token sequence length is greater than the minimum
    if num_tokens_current_chunk >= min_seq_len:
        ticks_split.append((bar_ticks[bi_start_chunk], score.end() + 1))

    if len(ticks_split) == 1:
        return [score]

    return [
        score.clip(t_start, t_end).shift_time(-t_start)
        for t_start, t_end in ticks_split
    ]


def get_average_num_tokens_per_note(
    tokenizer: MusicTokenizer, files_paths: Sequence[Path]
) -> float:
    """
    Return the average number of tokens per note (tpn) for a list of music files.

    With a trained tokenizer, the average tpn is likely to be very low.

    :param tokenizer: tokenizer.
    :param files_paths: list of paths to music files.
    :return: the average tokens per note.
    """
    num_tokens_per_note = []
    for file_path in files_paths:
        try:
            score = Score(file_path)
        except SCORE_LOADING_EXCEPTION:
            continue
        tok_seq = tokenizer(score)
        if tokenizer.one_token_stream:
            if (num_notes := score.note_num()) > 0:
                num_tokens_per_note.append(len(tok_seq) / num_notes)
        else:
            for track, seq in zip(score.tracks, tok_seq):
                if (num_notes := track.note_num()) > 0:
                    num_tokens_per_note.append(len(seq) / num_notes)

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
        sub_seq.append(seq[i : i + max_seq_len])
        i += len(sub_seq[-1])  # could be replaced with max_seq_len

    return sub_seq


def split_tokens_files_to_subsequences(
    files_paths: Sequence[Path],
    out_dir: Path,
    min_seq_len: int,
    max_seq_len: int,
    one_token_stream: bool = True,
) -> None:
    """
    Split JSON tokens files into subsequences of defined lengths.

    This method is particularly useful if you plan to use a
    :class:`miditok.pytorch_data.DatasetJSON`, as it would split token sequences
    into subsequences with the desired lengths before loading them for training.

    :param files_paths: list of files of tokens to split.
    :param out_dir: output directory to save the subsequences.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :param one_token_stream: give False if the token files contains multiple tracks,
        i.e. the first dimension of the value of the "ids" entry corresponds to several
        tracks. Otherwise, leave False. (default: True)
    """
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
