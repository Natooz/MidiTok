#!/usr/bin/python3 python

"""Test methods."""

from __future__ import annotations

from copy import copy
from pathlib import Path

import pytest
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
)

from miditok import REMI, TokenizerConfig
from miditok.constants import CLASS_OF_INST
from miditok.utils import (
    merge_same_program_tracks,
    merge_tracks,
    merge_tracks_per_class,
    num_bar_pos,
    remove_duplicated_notes,
)

from .utils import (
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
    TOKENIZER_CONFIG_KWARGS,
    check_midis_equals,
    del_invalid_time_sig,
)


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


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
def test_check_midi_equals(midi_path: Path):
    midi = Score(midi_path)
    midi_copy = copy(midi)

    # Check when midi is untouched
    assert check_midis_equals(midi, midi_copy)

    # Altering notes
    i = 0
    while i < len(midi_copy.tracks):
        if len(midi_copy.tracks[i].notes) > 0:
            midi_copy.tracks[i].notes[-1].pitch += 5
            assert not check_midis_equals(midi, midi_copy)
            break
        i += 1

    # Altering track events
    if len(midi_copy.tracks) > 0:
        # Altering pedals
        midi_copy = copy(midi)
        if len(midi_copy.tracks[0].pedals) == 0:
            midi_copy.tracks[0].pedals.append(Pedal(0, 10))
        else:
            midi_copy.tracks[0].pedals[-1].duration += 10
        assert not check_midis_equals(midi, midi_copy)

        # Altering pitch bends
        midi_copy = copy(midi)
        if len(midi_copy.tracks[0].pitch_bends) == 0:
            midi_copy.tracks[0].pitch_bends.append(PitchBend(50, 10))
        else:
            midi_copy.tracks[0].pitch_bends[-1].value += 10
        assert not check_midis_equals(midi, midi_copy)

    # Altering tempos
    midi_copy = copy(midi)
    if len(midi_copy.tempos) == 0:
        midi_copy.tempos.append(Tempo(50, 10))
    else:
        midi_copy.tempos[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)

    # Altering time signatures
    midi_copy = copy(midi)
    if len(midi_copy.time_signatures) == 0:
        midi_copy.time_signatures.append(TimeSignature(10, 4, 4))
    else:
        midi_copy.time_signatures[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)


def test_merge_tracks(
    midi_path: str | Path = MIDI_PATHS_ONE_TRACK[0],
):
    # Load MIDI and only keep the first track
    midi = Score(midi_path)
    midi.tracks = [midi.tracks[0]]

    # Duplicate the track and merge it
    original_track = copy(midi.tracks[0])
    midi.tracks.append(copy(midi.tracks[0]))

    # Test merge with effects
    merge_tracks(midi, effects=True)
    assert len(midi.tracks[0].notes) == 2 * len(original_track.notes)
    assert len(midi.tracks[0].pedals) == 2 * len(original_track.pedals)
    assert len(midi.tracks[0].controls) == 2 * len(original_track.controls)
    assert len(midi.tracks[0].pitch_bends) == 2 * len(original_track.pitch_bends)


@pytest.mark.parametrize("midi_path", MIDI_PATHS_MULTITRACK)
def test_merge_same_program_tracks_and_by_class(midi_path: str | Path):
    midi = Score(midi_path)
    for track in midi.tracks:
        if track.is_drum:
            track.program = 128

    # Test merge same program
    midi_copy = copy(midi)
    programs = [track.program for track in midi_copy.tracks]
    unique_programs = list(set(programs))
    merge_same_program_tracks(midi_copy.tracks)
    new_programs = [track.program for track in midi_copy.tracks]
    unique_programs.sort()
    new_programs.sort()
    assert new_programs == unique_programs

    # Test merge same class
    midi_copy = copy(midi)
    merge_tracks_per_class(
        midi_copy,
        CLASS_OF_INST,
        valid_programs=list(range(-1, 129)),
        filter_pitches=True,
    )


def test_num_pos():
    (tok_conf := TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)).use_time_signatures = True
    tokenizer = REMI(tok_conf)
    midi = Score(MIDI_PATHS_ONE_TRACK[0])
    del_invalid_time_sig(midi.time_signatures, tokenizer.time_signatures)
    _ = num_bar_pos(
        tokenizer.midi_to_tokens(midi)[0].ids,
        tokenizer["Bar_None"],
        tokenizer.token_ids_of_type("Position"),
    )


def test_remove_duplicated_notes():
    sets = [
        # No duplicated
        (
            [
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=10, pitch=51, velocity=50),
                Note(time=2, duration=10, pitch=50, velocity=50),
                Note(time=4, duration=10, pitch=50, velocity=50),
            ],
            0,
            0,
        ),
        # One duplicated with dur
        (
            [
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=2, duration=10, pitch=50, velocity=50),
                Note(time=4, duration=10, pitch=50, velocity=50),
            ],
            1,
            1,
        ),
        (
            [
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=11, pitch=50, velocity=50),
                Note(time=2, duration=10, pitch=50, velocity=50),
                Note(time=4, duration=10, pitch=50, velocity=50),
            ],
            1,
            0,
        ),
        (
            [
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=11, pitch=50, velocity=50),
                Note(time=2, duration=10, pitch=50, velocity=50),
                Note(time=4, duration=10, pitch=50, velocity=50),
            ],
            2,
            1,
        ),
        (
            [
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=10, pitch=50, velocity=50),
                Note(time=0, duration=11, pitch=50, velocity=50),
                Note(time=2, duration=10, pitch=50, velocity=50),
                Note(time=2, duration=11, pitch=50, velocity=50),
                Note(time=4, duration=10, pitch=50, velocity=50),
            ],
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
