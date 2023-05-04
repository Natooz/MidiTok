#!/usr/bin/python3 python

"""Test methods

"""

from pathlib import Path
from copy import deepcopy

from miditoolkit import MidiFile

from miditok import REMI
from miditok.utils import (
    merge_tracks,
    merge_tracks_per_class,
    merge_same_program_tracks,
    nb_bar_pos,
)
from miditok.constants import CLASS_OF_INST


def test_merge_tracks():
    midi = MidiFile(Path("tests", "Maestro_MIDIs", "Maestro_1.mid"))
    original_track = deepcopy(midi.instruments[0])
    midi.instruments.append(deepcopy(midi.instruments[0]))
    merge_tracks(midi.instruments)
    assert len(midi.instruments[0].notes) == 2 * len(original_track.notes)


def test_merge_same_program_tracks_and_by_class():
    multitrack_midi_paths = list(Path("tests", "Multitrack_MIDIs").glob("**/*.mid"))
    for midi_path in multitrack_midi_paths:
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
        tokenizer(Path("tests", "Maestro_MIDIs", "Maestro_1.mid"))[0].ids,
        tokenizer["Bar_None"],
        tokenizer.token_ids_of_type("Position"),
    )


if __name__ == "__main__":
    test_merge_tracks()
    test_merge_same_program_tracks_and_by_class()
    test_nb_pos()
