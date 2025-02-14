"""Data augmentation methods."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2

import numpy as np
from symusic import Note, Score
from tqdm import tqdm

from miditok.constants import (
    MIDI_FILES_EXTENSIONS,
    MIDI_INSTRUMENTS,
    SCORE_LOADING_EXCEPTION,
    SUPPORTED_MUSIC_FILE_EXTENSIONS,
)


def augment_dataset(
    data_path: Path | str,
    pitch_offsets: list[int] | None = None,
    velocity_offsets: list[int] | None = None,
    duration_offsets: list[int] | None = None,
    all_offset_combinations: bool = False,
    restrict_on_program_tessitura: bool = True,
    velocity_range: tuple[int, int] = (1, 127),
    duration_in_ticks: bool = False,
    min_duration: int | float = 0.03125,
    out_path: Path | str | None = None,
    copy_original_in_new_location: bool = True,
    save_data_aug_report: bool = True,
) -> None:
    r"""
    Perform data augmentation on a dataset of music files.

    The new created files have names in two parts, separated with a "#" character. Make
    sure your files do not have '§' in their names if you intend to reuse the
    information of the second part in some script.
    **Drum tracks are not augmented.**.

    :param data_path: root path to the folder containing tokenized json files.
    :param pitch_offsets: list of pitch offsets for augmentation. (default: ``None``)
    :param velocity_offsets: list of velocity offsets for augmentation. If you plan to
        tokenize these files, the velocity offsets should be chosen accordingly to the
        number of velocities in your tokenizer's vocabulary (``num_velocities``).
        (default: ``None``)
    :param duration_offsets: list of duration offsets for augmentation, to be given
        either in beats if ``duration_in_ticks`` is ``False``, in ticks otherwise.
        (default: ``None``)
    :param all_offset_combinations: will perform data augmentation on all the possible
        combinations of offset values. If set to ``False``, the method will only
        augment on the offsets separately without combining them.
    :param restrict_on_program_tessitura: if ``True``, the method will consider the
        recommended pitch values of each instrument/program as the range of possible
        values after augmentation. Otherwise, the ``(0, 127)`` range will be used.
        (default: ``True``)
    :param velocity_range: minimum and maximum velocity values. (default: ``(1, 127)``)
    :param duration_in_ticks: if given ``True``, the ``duration_offset`` argument will
        be considered as expressed in ticks. Otherwise, it is considered in beats, and
        the equivalent in ticks will be determined by multiplying it by the MIDI's
        time division (480 by default for abc files). (default: False)
    :param min_duration: minimum duration limit to apply if ``duration_offset`` is
        negative. If ``duration_in_ticks`` is ``True``, it must be given in ticks,
        otherwise in beats as a float or integer. (default: 0.03125)
    :param out_path: output path to save the augmented files. Original (non-augmented)
        files will be saved to this location. If none is given, they will be saved in
        the same location as the data_path. (default: None)
    :param copy_original_in_new_location: if given True, the original (non-augmented)
        files will be saved in the out_path location too. (default: True)
    :param save_data_aug_report: will save numbers from the data augmentation in a
        ``data_augmentation_report.txt`` file in the output directory. (default: True)
    """
    if out_path is None:
        out_path = Path(data_path)
    else:
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
    files_paths = [
        path
        for path in data_path.glob("**/*")
        if path.suffix in SUPPORTED_MUSIC_FILE_EXTENSIONS
    ]

    num_augmentations, num_tracks_augmented = 0, 0
    for file_path in tqdm(files_paths, desc="Performing data augmentation"):
        try:
            score = Score(file_path)
        except SCORE_LOADING_EXCEPTION:
            continue

        augmented_scores = augment_score_multiple_offsets(
            score,
            pitch_offsets,
            velocity_offsets,
            duration_offsets,
            all_offset_combinations,
            restrict_on_program_tessitura,
            velocity_range,
            duration_in_ticks,
            min_duration,
        )
        for aug_offsets, score_aug in augmented_scores:
            if len(score_aug.tracks) == 0:
                continue
            suffix = "#" + "_".join(
                [
                    f"{t}{offset}"
                    for t, offset in zip(["p", "v", "d"], aug_offsets)
                    if offset != 0
                ]
            )
            saving_path = out_path / file_path.parent.relative_to(data_path)
            saving_path.mkdir(parents=True, exist_ok=True)
            saving_path /= f"{file_path.stem}{suffix}{file_path.suffix}"
            if file_path.suffix in MIDI_FILES_EXTENSIONS:
                score_aug.dump_midi(saving_path)
            else:
                score_aug.dump_abc(saving_path)
            num_augmentations += 1
            num_tracks_augmented += len(score_aug.tracks)
        if copy_original_in_new_location and out_path != data_path:
            saving_path = out_path / file_path.relative_to(data_path)
            saving_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(file_path, saving_path)

    # Saves data augmentation report, json encoded with txt extension to not mess with
    # others json files
    if save_data_aug_report:
        with Path.open(out_path / "data_augmentation_report.txt", "w") as outfile:
            json.dump(
                {
                    "pitch_offsets": pitch_offsets,
                    "velocity_offsets": velocity_offsets,
                    "duration_offsets": duration_offsets,
                    "all_offset_combinations": all_offset_combinations,
                    "velocity_range": velocity_range,
                    "min_duration": min_duration,
                    "num_tracks_augmented": num_tracks_augmented,
                    "num_files_before": len(files_paths),
                    "num_files_after": len(files_paths) + num_augmentations,
                },
                outfile,
            )


def _filter_offset_tuples_to_score(
    pitch_offsets: list[int],
    score: Score,
    restrict_on_program_tessitura: bool,
) -> list[int]:
    r"""
    Remove pitch offset values that would cause errors or are out of tessitura.

    :param pitch_offsets: list of pitch offsets for augmentation.
    :param score: ``symusic.Score`` object to augment (default: None)
    :param restrict_on_program_tessitura: if ``True``, the method will consider the
        recommended pitch values of each instrument/program as the range of possible
        values after augmentation. Otherwise, the ``(0, 127)`` range will be used.
    :return: the filtered offsets of pitch.
    """
    # Get min and max pitches in the Score (except drum tracks)
    all_pitches = [
        track.notes.numpy()["pitch"]
        for track in score.tracks
        if not track.is_drum and len(track.notes) > 0
    ]
    min_pitches = [np.min(pitches) for pitches in all_pitches]
    max_pitches = [np.max(pitches) for pitches in all_pitches]

    # Determine the minimum and maximum possible pitch offsets
    if restrict_on_program_tessitura:
        min_possible_pitch_offset, max_possible_pitch_offset = -127, 127
        for min_pitch, max_pitch, track in zip(
            min_pitches, max_pitches, [t for t in score.tracks if not t.is_drum]
        ):
            pitch_range = MIDI_INSTRUMENTS[track.program]["pitch_range"]
            min_possible_pitch_offset = max(
                min_possible_pitch_offset, pitch_range.start - min_pitch
            )
            max_possible_pitch_offset = min(
                max_possible_pitch_offset, pitch_range.stop - 1 - max_pitch
            )
    else:
        min_possible_pitch_offset = -min(min_pitches)
        max_possible_pitch_offset = 127 - max(max_pitches)

    return [
        pitch_offset
        for pitch_offset in pitch_offsets
        if min_possible_pitch_offset <= pitch_offset <= max_possible_pitch_offset
    ]


def _create_offsets_tuples(
    score: Score,
    pitch_offsets: list[int] | None = None,
    velocity_offsets: list[int] | None = None,
    duration_offsets: list[int] | None = None,
    all_offset_combinations: bool = False,
    restrict_on_program_tessitura: bool = True,
) -> list[tuple[int, int, int]]:
    """
    Create the data augmentation tuples combinations from lists of offsets.

    :param score: ``symusic.Score`` object to augment.
    :param pitch_offsets: list of pitch offsets for augmentation.
    :param velocity_offsets: list of velocity offsets for augmentation. If you plan to
        tokenize this file, the velocity offsets should be chosen accordingly to the
        number of velocities in your tokenizer's vocabulary (``num_velocities``).
    :param duration_offsets: list of duration offsets for augmentation, to be given
        either in beats if ``duration_in_ticks`` is ``False``, in ticks otherwise.
    :param all_offset_combinations: will perform data augmentation on all the possible
        combinations of offset values. If set to ``False``, the method will only
        augment on the offsets separately without combining them.
    :param restrict_on_program_tessitura: if ``True``, the method will consider the
        recommended pitch values of each instrument/program as the range of possible
        values after augmentation. Otherwise, the ``(0, 127)`` range will be used.
        (default: ``True``)
    :return:
    """
    # Remove pitch offsets that would cause errors or are out of tessitura
    pitch_offsets = _filter_offset_tuples_to_score(
        pitch_offsets, score, restrict_on_program_tessitura
    )
    # Create basic offsets
    offsets = [(pitch_offset, 0, 0) for pitch_offset in pitch_offsets]
    offsets += [(0, velocity_offset, 0) for velocity_offset in velocity_offsets]
    offsets += [(0, 0, duration_offset) for duration_offset in duration_offsets]

    # Adds all possible combinations
    if all_offset_combinations:
        for idx, offsets_type in enumerate(
            (pitch_offsets, velocity_offsets, duration_offsets)
        ):
            if offsets_type is None:
                continue
            for offset_val in offsets_type:
                for offset_idx in range(len(offsets)):
                    if offsets[offset_idx][idx] != offset_val:
                        (new_offset := list(offsets[offset_idx]))[idx] = offset_val
                        new_offset = tuple(new_offset)
                        # This could be optimized by removing the "in" and parametrize
                        if new_offset not in offsets:
                            offsets.append(new_offset)

    return offsets


def augment_score(
    score: Score,
    pitch_offset: int = 0,
    velocity_offset: int = 0,
    duration_offset: int | float = 0,
    velocity_range: tuple[int, int] = (1, 127),
    duration_in_ticks: bool = False,
    min_duration: int | float = 0.03125,
    augment_copy: bool = True,
) -> Score:
    r"""
    Augment a Score object by shifting its pitch, velocity and/or duration values.

    Velocity and duration values will be clipped according to the ``velocity_range``
    and ``min_duration`` arguments. Drum tracks are only augmented on the velocity.
    If you are using a pitch offset, make sure the files doesn't contain notes with
    pitches that would end outside the conventional ``(0, 127)`` range, the method will
    otherwise crash.

    :param score: ``symusic.Score`` object to augment.
    :param pitch_offset: pitch offset for augmentation. (default: ``0``)
    :param velocity_offset: velocity offset for augmentation. If you plan to tokenize
        this file, the velocity offset should be chosen accordingly to the number of
        velocities in your tokenizer's vocabulary (``num_velocities``).
        (default: ``0``)
    :param duration_offset: duration offset for augmentation, to be given
        either in beats if ``duration_in_ticks`` is ``False``, in ticks otherwise.
        (default: ``0``)
    :param velocity_range: minimum and maximum velocity values. (default: ``(1, 127)``)
    :param duration_in_ticks: if given ``True``, the ``duration_offset`` argument will
        be considered as expressed in ticks. Otherwise, it is considered in beats, and
        the equivalent in ticks will be determined by multiplying it by the MIDI's
        time division (480 by default for abc files). (default: ``False``)
    :param min_duration: minimum duration limit to apply if ``duration_offset`` is
        negative. If ``duration_in_ticks`` is ``True``, it must be given in ticks,
        otherwise in beats as a float or integer. (default: ``0.03125``)
    :param augment_copy: if given True, a copy of the input ``symusic.Score`` object is
        augmented and returned. If False, the input ``symusic.Score`` object is modified
        in-place. (default: ``True``)
    :return: the augmented ``symusic.Score`` object.
    """
    score_aug = score.copy() if augment_copy else score
    # score_aug = score.copy(deep=True)

    if pitch_offset != 0:
        for track in score_aug.tracks:
            if not track.is_drum:
                track.shift_pitch(pitch_offset, inplace=True)

    if velocity_offset != 0:
        for track in score_aug.tracks:
            for note in track.notes:
                vel_shifted = note.velocity + velocity_offset
                if velocity_offset < 0:
                    note.velocity = max(min(velocity_range), vel_shifted)
                else:
                    note.velocity = min(max(velocity_range), vel_shifted)

    if duration_offset != 0:
        if not duration_in_ticks:
            duration_offset = max(round(duration_offset * score.ticks_per_quarter), 1)
            min_duration = max(round(min_duration * score.ticks_per_quarter), 1)
        # If in ticks but the offset is a float, we round it to the closest integer.
        elif isinstance(duration_offset, float):
            duration_offset = round(duration_offset)
            min_duration = round(min_duration)

        for track in score_aug.tracks:
            if track.is_drum:
                continue
            notes_soa = track.notes.numpy()
            notes_soa["duration"] += duration_offset
            if duration_offset < 0:
                # This will also set durations that where originally between 0 and
                # min_duration to min_duration
                mask = np.where(notes_soa["duration"] < min_duration)[0]
                notes_soa["duration"][mask] = min_duration

            track.notes = Note.from_numpy(**notes_soa)

    return score_aug


def augment_score_multiple_offsets(
    score: Score,
    pitch_offsets: list[int] | None = None,
    velocity_offsets: list[int] | None = None,
    duration_offsets: list[int] | None = None,
    all_offset_combinations: bool = False,
    restrict_on_program_tessitura: bool = True,
    velocity_range: tuple[int, int] = (1, 127),
    duration_in_ticks: bool = False,
    min_duration: int | float = 0.03125,
) -> list[tuple[tuple[int, int, int], Score]]:
    r"""
    Perform data augmentation on a ``symusic.Score`` object with multiple offset values.

    Velocity and duration values will be clipped according to the ``velocity_range`` and
    ``min_duration`` arguments. Drum tracks are only augmented on the velocity.

    :param score: ``symusic.Score`` object to augment.
    :param pitch_offsets: list of pitch offsets for augmentation.
    :param velocity_offsets: list of velocity offsets for augmentation. If you plan to
        tokenize this file, the velocity offsets should be chosen accordingly to the
        number of velocities in your tokenizer's vocabulary (``num_velocities``).
        (default: ``None``)
    :param duration_offsets: list of duration offsets for augmentation, to be given
        either in beats if ``duration_in_ticks`` is ``False``, in ticks otherwise.
        (default: ``None``)
    :param all_offset_combinations: will perform data augmentation on all the possible
        combinations of offset values. If set to ``False``, the method will only
        augment on the offsets separately without combining them. (default: ``None``)
    :param restrict_on_program_tessitura: if ``True``, the method will consider the
        recommended pitch values of each instrument/program as the range of possible
        values after augmentation. Otherwise, the ``(0, 127)`` range will be used.
        (default: ``True``)
    :param velocity_range: minimum and maximum velocity values. (default: ``(1, 127)``)
    :param duration_in_ticks: if given ``True``, the ``duration_offset`` argument will
        be considered as expressed in ticks. Otherwise, it is considered in beats, and
        the equivalent in ticks will be determined by multiplying it by the MIDI's
        time division (480 by default for abc files). (default: False)
    :param min_duration: minimum duration limit to apply if ``duration_offset`` is
        negative. If ``duration_in_ticks`` is ``True``, it must be given in ticks,
        otherwise in beats as a float or integer. (default: 0.03125)
    :return: augmented ``symusic.Score`` objects.
    """
    # Create offset tuples
    # If duration offsets are given in beats, we convert them to ticks here so
    # that the conversion is not done multiple times in downstream methods
    if duration_offsets and not duration_in_ticks:
        duration_offsets = [
            round(duration_offset * score.ticks_per_quarter)
            for duration_offset in duration_offsets
        ]
        min_duration = max(round(min_duration * score.ticks_per_quarter), 1)
    offsets = _create_offsets_tuples(
        score,
        pitch_offsets,
        velocity_offsets,
        duration_offsets,
        all_offset_combinations,
        restrict_on_program_tessitura,
    )

    # Create augmented versions
    return [
        (
            offsets_tuple,
            augment_score(
                score,
                *offsets_tuple,
                velocity_range=velocity_range,
                duration_in_ticks=True,
                min_duration=min_duration,
            ),
        )
        for offsets_tuple in offsets
    ]
