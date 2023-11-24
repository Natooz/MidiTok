#!/usr/bin/python3 python

"""Test methods

"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import pytest
from miditoolkit import MidiFile

from miditok import REMI
from miditok.constants import CLASS_OF_INST
from miditok.utils import (
    merge_same_program_tracks,
    merge_tracks,
    merge_tracks_per_class,
    nb_bar_pos,
)

from .utils import MIDI_PATHS_MULTITRACK, MIDI_PATHS_ONE_TRACK


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
