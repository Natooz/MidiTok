"""Test methods."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING

import pytest
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TextMeta,
    TimeSignature,
)
from symusic.core import NoteTickList

import miditok.utils.utils
from miditok import REMI, TokenizerConfig
from miditok.constants import CLASS_OF_INST
from miditok.utils import (
    merge_same_program_tracks,
    merge_tracks,
    merge_tracks_per_class,
    num_bar_pos,
    remove_duplicated_notes,
)

from .utils_tests import (
    MIDI_PATHS_CORRUPTED,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
    TEST_LOG_DIR,
    TOKENIZER_CONFIG_KWARGS,
    check_scores_equals,
    del_invalid_time_sig,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_containers_assertions():
    note1 = [Note(0, 30, 50, 120), Note(0, 30, 49, 120)]
    note2 = [Note(0, 30, 51, 120), Note(0, 30, 49, 120)]
    note3 = [Note(0, 30, 51, 120), Note(0, 30, 49, 120)]
    assert note1 != note2
    assert note2 == note3

    tc1 = [Tempo(2, 120), Tempo(0, 110)]
    tc2 = [Tempo(3, 120), Tempo(0, 110)]
    tc3 = [Tempo(3, 120), Tempo(0, 110)]
    assert tc1 != tc2
    assert tc2 == tc3

    ts1 = [TimeSignature(0, 4, 4), TimeSignature(10, 6, 4)]
    ts2 = [TimeSignature(0, 2, 4), TimeSignature(10, 6, 4)]
    ts3 = [TimeSignature(0, 2, 4), TimeSignature(10, 6, 4)]
    assert ts1 != ts2
    assert ts2 == ts3

    sp1 = [Pedal(0, 2), Pedal(10, 20)]
    sp2 = [Pedal(1, 2), Pedal(15, 20)]
    sp3 = [Pedal(1, 2), Pedal(15, 20)]
    assert sp1 != sp2
    assert sp2 == sp3

    pb1 = [PitchBend(2, 120), PitchBend(0, 110)]
    pb2 = [PitchBend(3, 120), PitchBend(0, 110)]
    pb3 = [PitchBend(3, 120), PitchBend(0, 110)]
    assert pb1 != pb2
    assert pb2 == pb3

    ks1 = [KeySignature(2, 0, 1), KeySignature(0, 0, 1)]
    ks2 = [KeySignature(20, 0, 1), KeySignature(0, 0, 1)]
    ks3 = [KeySignature(20, 0, 1), KeySignature(0, 0, 1)]
    assert ks1 != ks2
    assert ks2 == ks3

    cc1 = [ControlChange(2, 120, 50), ControlChange(0, 110, 50)]
    cc2 = [ControlChange(2, 120, 50), ControlChange(10, 110, 50)]
    cc3 = [ControlChange(2, 120, 50), ControlChange(10, 110, 50)]
    assert cc1 != cc2
    assert cc2 == cc3


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK, ids=lambda p: p.name)
def test_check_scores_equals(file_path: Path):
    score = Score(file_path)
    score_copy = score.copy()
    # score_copy = score.copy(deep=True)

    # Check when midi is untouched
    assert check_scores_equals(score, score_copy)

    # Altering notes
    i = 0
    while i < len(score_copy.tracks):
        if len(score_copy.tracks[i].notes) > 0:
            score_copy.tracks[i].notes[-1].pitch += 5
            assert not check_scores_equals(score, score_copy)
            break
        i += 1

    # Altering track events
    if len(score_copy.tracks) > 0:
        # Altering pedals
        score_copy = score.copy()
        # score_copy = score.copy(deep=True)
        if len(score_copy.tracks[0].pedals) == 0:
            score_copy.tracks[0].pedals.append(Pedal(0, 10))
        else:
            score_copy.tracks[0].pedals[-1].duration += 10
        assert not check_scores_equals(score, score_copy)

        # Altering pitch bends
        score_copy = score.copy()
        # score_copy = score.copy(deep=True)
        if len(score_copy.tracks[0].pitch_bends) == 0:
            score_copy.tracks[0].pitch_bends.append(PitchBend(50, 10))
        else:
            score_copy.tracks[0].pitch_bends[-1].value += 10
        assert not check_scores_equals(score, score_copy)

    # Altering tempos
    score_copy = score.copy()
    # score_copy = score.copy(deep=True)
    if len(score_copy.tempos) == 0:
        score_copy.tempos.append(Tempo(50, 10))
    else:
        score_copy.tempos[-1].time += 10
    assert not check_scores_equals(score, score_copy)

    # Altering time signatures
    score_copy = score.copy()
    # score_copy = score.copy(deep=True)
    if len(score_copy.time_signatures) == 0:
        score_copy.time_signatures.append(TimeSignature(10, 4, 4))
    else:
        score_copy.time_signatures[-1].time += 10
    assert not check_scores_equals(score, score_copy)


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK[:1], ids=lambda p: p.name)
def test_merge_tracks(file_path: str | Path):
    # Load music file and only keep the first track
    score = Score(file_path)
    score.tracks = [score.tracks[0]]

    # Duplicate the track and merge it
    original_track = copy(score.tracks[0])
    score.tracks.append(copy(score.tracks[0]))

    # Test merge with effects
    merge_tracks(score, effects=True)
    assert len(score.tracks[0].notes) == 2 * len(original_track.notes)
    assert len(score.tracks[0].pedals) == 2 * len(original_track.pedals)
    assert len(score.tracks[0].controls) == 2 * len(original_track.controls)
    assert len(score.tracks[0].pitch_bends) == 2 * len(original_track.pitch_bends)


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
def test_merge_same_program_tracks_and_by_class(file_path: str | Path):
    score = Score(file_path)
    for track in score.tracks:
        if track.is_drum:
            track.program = 128

    # Test merge same program
    score_copy = copy(score)
    programs = [track.program for track in score_copy.tracks]
    unique_programs = list(set(programs))
    merge_same_program_tracks(score_copy.tracks)
    new_programs = [track.program for track in score_copy.tracks]
    unique_programs.sort()
    new_programs.sort()
    assert new_programs == unique_programs

    # Test merge same class
    score_copy = copy(score)
    merge_tracks_per_class(
        score_copy,
        CLASS_OF_INST,
        valid_programs=list(range(-1, 129)),
        filter_pitches=True,
    )


def test_num_pos():
    (tok_conf := TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)).use_time_signatures = True
    tokenizer = REMI(tok_conf)
    score = Score(MIDI_PATHS_ONE_TRACK[0])
    del_invalid_time_sig(score.time_signatures, tokenizer.time_signatures)
    _ = num_bar_pos(
        tokenizer.encode(score)[0].ids,
        tokenizer["Bar_None"],
        tokenizer.token_ids_of_type("Position"),
    )


def test_remove_duplicated_notes():
    sets = [
        # No duplicated
        (
            NoteTickList(
                [
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=10, pitch=51, velocity=50),
                    Note(time=2, duration=10, pitch=50, velocity=50),
                    Note(time=4, duration=10, pitch=50, velocity=50),
                ]
            ),
            0,
            0,
        ),
        # One duplicated with dur
        (
            NoteTickList(
                [
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=2, duration=10, pitch=50, velocity=50),
                    Note(time=4, duration=10, pitch=50, velocity=50),
                ]
            ),
            1,
            1,
        ),
        (
            NoteTickList(
                [
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=11, pitch=50, velocity=50),
                    Note(time=2, duration=10, pitch=50, velocity=50),
                    Note(time=4, duration=10, pitch=50, velocity=50),
                ]
            ),
            1,
            0,
        ),
        (
            NoteTickList(
                [
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=11, pitch=50, velocity=50),
                    Note(time=2, duration=10, pitch=50, velocity=50),
                    Note(time=4, duration=10, pitch=50, velocity=50),
                ]
            ),
            2,
            1,
        ),
        (
            NoteTickList(
                [
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=10, pitch=50, velocity=50),
                    Note(time=0, duration=11, pitch=50, velocity=50),
                    Note(time=2, duration=10, pitch=50, velocity=50),
                    Note(time=2, duration=11, pitch=50, velocity=50),
                    Note(time=4, duration=10, pitch=50, velocity=50),
                ]
            ),
            3,
            1,
        ),
    ]

    for notes, diff, diff_with_duration in sets:
        remove_duplicated_notes(notes_filtered := copy(notes))
        remove_duplicated_notes(
            notes_filtered_dur := copy(notes), consider_duration=True
        )

        if diff == 0:
            assert notes == notes_filtered
        else:
            assert len(notes) - len(notes_filtered) == diff
        if diff_with_duration == 0:
            assert notes == notes_filtered_dur
        else:
            assert len(notes) - len(notes_filtered_dur) == diff_with_duration


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK, ids=lambda p: p.name)
def test_get_bars(file_path: Path):
    save_bars_markers = False
    # Used for debug, this method do not make assertions
    score = Score(file_path)
    bars_ticks = miditok.utils.get_bars_ticks(score)
    if save_bars_markers:
        for bar_num, bar_tick in enumerate(bars_ticks):
            score.markers.append(TextMeta(bar_tick, f"Bar {bar_num + 1}"))
        score.dump_midi(TEST_LOG_DIR / f"{file_path.stem}_bars.mid")


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
def test_get_num_notes_per_bar(file_path: Path):
    score = Score(file_path)
    num_notes = miditok.utils.get_num_notes_per_bar(score)
    num_notes_track_indep = miditok.utils.get_num_notes_per_bar(
        score, tracks_indep=True
    )
    num_notes_track_indep_summed = [sum(num_n) for num_n in num_notes_track_indep]
    assert num_notes == num_notes_track_indep_summed


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
@pytest.mark.parametrize("max_num_beats", [16], ids=lambda x: f"{x} max beats")
def test_split_concat_score(file_path: Path, max_num_beats: int):
    score = Score(file_path)
    score_splits = miditok.utils.split_score_per_beats(score, max_num_beats)
    ticks_beat = miditok.utils.get_beats_ticks(score, only_notes_onsets=True)

    # Check there is the good number of split music files
    assert len(score_splits) == ceil(len(ticks_beat) / max_num_beats)

    """# Saves each chunk separately (for debug purposes)
    from tests.utils_tests import HERE
    for i, score_split in enumerate(score_splits):
        score_split.dump_midi(HERE / "score_splits" / f"{i}.mid")"""

    # Concatenate split MIDIs and assert its equal to original one
    end_ticks = [
        ticks_beat[i] for i in range(max_num_beats, len(ticks_beat), max_num_beats)
    ]
    score_concat = miditok.utils.concat_scores(score_splits, end_ticks)

    # Assert the concatenated Score is identical to the original one
    # We do not test tempos, time signatures and key signature as they are duplicated
    # in score_concat (same consecutive ones for each chunk).
    assert score.tracks == score_concat.tracks
    assert score.markers == score_concat.markers


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
def test_split_score_per_tracks(file_path: Path):
    score = Score(file_path)
    score_splits = miditok.utils.split_score_per_tracks(score)

    # Check there is the good number of split Scores
    assert len(score_splits) == len(score.tracks)

    # Merge split Scores and assert its equal to original one
    for score_split in score_splits[1:]:  # dedup global events
        score_split.tempos = []
        score_split.time_signatures = []
        score_split.key_signatures = []
        score_split.markers = []
    score_merged = miditok.utils.merge_scores(score_splits)

    # Assert the merges Score is identical to the original one
    assert score == score_merged


def test_filter_dataset():
    files_paths = MIDI_PATHS_MULTITRACK + MIDI_PATHS_CORRUPTED
    assert miditok.utils.filter_dataset(files_paths) == MIDI_PATHS_MULTITRACK
