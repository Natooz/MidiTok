#!/usr/bin/python3 python

"""Test methods

"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import pytest
from miditoolkit import (
    ControlChange,
    KeySignature,
    MidiFile,
    Pedal,
    PitchBend,
    TempoChange,
    TimeSignature,
)

from miditok import REMI
from miditok.constants import CLASS_OF_INST
from miditok.utils import (
    merge_same_program_tracks,
    merge_tracks,
    merge_tracks_per_class,
    nb_bar_pos,
)

from .utils import MIDI_PATHS_MULTITRACK, MIDI_PATHS_ONE_TRACK, check_midis_equals


def test_containers_assertions():
    tc1 = [TempoChange(120, 2), TempoChange(110, 0)]
    tc2 = [TempoChange(120, 3), TempoChange(110, 0)]
    tc3 = [TempoChange(120, 3), TempoChange(110, 0)]
    assert tc1 != tc2
    assert tc2 == tc3

    ts1 = [TimeSignature(4, 4, 0), TimeSignature(6, 4, 10)]
    ts2 = [TimeSignature(2, 4, 0), TimeSignature(6, 4, 10)]
    ts3 = [TimeSignature(2, 4, 0), TimeSignature(6, 4, 10)]
    assert ts1 != ts2
    assert ts2 == ts3

    sp1 = [Pedal(0, 2), TempoChange(10, 20)]
    sp2 = [Pedal(0, 2), TempoChange(15, 20)]
    sp3 = [Pedal(0, 2), TempoChange(15, 20)]
    assert sp1 != sp2
    assert sp2 == sp3

    pb1 = [PitchBend(120, 2), PitchBend(110, 0)]
    pb2 = [PitchBend(120, 3), PitchBend(110, 0)]
    pb3 = [PitchBend(120, 3), PitchBend(110, 0)]
    assert pb1 != pb2
    assert pb2 == pb3

    ks1 = [KeySignature("C#", 2), KeySignature("C#", 0)]
    ks2 = [KeySignature("C#", 20), KeySignature("C#", 0)]
    ks3 = [KeySignature("C#", 20), KeySignature("C#", 0)]
    assert ks1 != ks2
    assert ks2 == ks3

    cc1 = [ControlChange(120, 50, 2), ControlChange(110, 50, 0)]
    cc2 = [ControlChange(120, 50, 2), ControlChange(110, 50, 10)]
    cc3 = [ControlChange(120, 50, 2), ControlChange(110, 50, 10)]
    assert cc1 != cc2
    assert cc2 == cc3


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
def test_check_midi_equals(midi_path: Path):
    midi = MidiFile(midi_path)
    midi_copy = deepcopy(midi)

    # Check when midi is untouched
    assert check_midis_equals(midi, midi_copy)[1]

    # Altering notes
    i = 0
    while i < len(midi_copy.instruments):
        if len(midi_copy.instruments[i].notes) > 0:
            midi_copy.instruments[i].notes[-1].pitch += 5
            assert not check_midis_equals(midi, midi_copy)[1]
            break
        i += 1

    # Altering pedals
    midi_copy = deepcopy(midi)
    if len(midi_copy.instruments[0].pedals) == 0:
        midi_copy.instruments[0].pedals.append(Pedal(0, 10))
    else:
        midi_copy.instruments[0].pedals[-1].end += 10
    assert not check_midis_equals(midi, midi_copy)[1]

    # Altering pitch bends
    midi_copy = deepcopy(midi)
    if len(midi_copy.instruments[0].pitch_bends) == 0:
        midi_copy.instruments[0].pitch_bends.append(PitchBend(50, 10))
    else:
        midi_copy.instruments[0].pitch_bends[-1].end += 10
    assert not check_midis_equals(midi, midi_copy)[1]

    # Altering tempos
    midi_copy = deepcopy(midi)
    if len(midi_copy.tempo_changes) == 0:
        midi_copy.tempo_changes.append(TempoChange(50, 10))
    else:
        midi_copy.tempo_changes[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)[1]

    # Altering time signatures
    midi_copy = deepcopy(midi)
    if len(midi_copy.time_signature_changes) == 0:
        midi_copy.time_signature_changes.append(TimeSignature(4, 4, 10))
    else:
        midi_copy.time_signature_changes[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)[1]


def test_merge_tracks(
    midi_path: Union[str, Path] = MIDI_PATHS_ONE_TRACK[0],
):
    # Load MIDI and only keep the first track
    midi = MidiFile(midi_path)
    midi.instruments = [midi.instruments[0]]

    # Duplicate the track and merge it
    original_track = deepcopy(midi.instruments[0])
    midi.instruments.append(deepcopy(midi.instruments[0]))

    # Test merge with effects
    merge_tracks(midi, effects=True)
    assert len(midi.instruments[0].notes) == 2 * len(original_track.notes)
    assert len(midi.instruments[0].pedals) == 2 * len(original_track.pedals)
    assert len(midi.instruments[0].control_changes) == 2 * len(
        original_track.control_changes
    )
    assert len(midi.instruments[0].pitch_bends) == 2 * len(original_track.pitch_bends)


@pytest.mark.parametrize("midi_path", MIDI_PATHS_MULTITRACK)
def test_merge_same_program_tracks_and_by_class(midi_path: Union[str, Path]):
    midi = MidiFile(midi_path)
    for track in midi.instruments:
        if track.is_drum:
            track.program = -1

    # Test merge same program
    midi_copy = deepcopy(midi)
    programs = [track.program for track in midi_copy.instruments]
    unique_programs = list(set(programs))
    merge_same_program_tracks(midi_copy.instruments)
    new_programs = [track.program for track in midi_copy.instruments]
    unique_programs.sort()
    new_programs.sort()
    assert new_programs == unique_programs

    # Test merge same class
    midi_copy = deepcopy(midi)
    merge_tracks_per_class(
        midi_copy,
        CLASS_OF_INST,
        valid_programs=list(range(-1, 128)),
        filter_pitches=True,
    )


def test_nb_pos():
    tokenizer = REMI()
    _ = nb_bar_pos(
        tokenizer(MIDI_PATHS_ONE_TRACK[0])[0].ids,
        tokenizer["Bar_None"],
        tokenizer.token_ids_of_type("Position"),
    )
