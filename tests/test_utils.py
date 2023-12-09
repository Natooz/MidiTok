#!/usr/bin/python3 python

"""Test methods

"""

from copy import copy
from pathlib import Path
from typing import Union

import pytest
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
)
from symusic.core import PedalTick

from miditok import REMI, TokenizerConfig
from miditok.constants import CLASS_OF_INST
from miditok.utils import (
    merge_same_program_tracks,
    merge_tracks,
    merge_tracks_per_class,
    nb_bar_pos,
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

    sp1 = [PedalTick(0, 2), PedalTick(10, 20)]
    sp2 = [PedalTick(1, 2), PedalTick(15, 20)]
    sp3 = [PedalTick(1, 2), PedalTick(15, 20)]
    assert sp1 != sp2
    assert sp2 == sp3

    pb1 = [PitchBend(2, 120), PitchBend(0, 110)]
    pb2 = [PitchBend(3, 120), PitchBend(0, 110)]
    pb3 = [PitchBend(3, 120), PitchBend(0, 110)]
    assert pb1 != pb2
    assert pb2 == pb3

    ks1 = [KeySignature(2, "C#"), KeySignature(0, "C#")]
    ks2 = [KeySignature(20, "C#"), KeySignature(0, "C#")]
    ks3 = [KeySignature(20, "C#"), KeySignature(0, "C#")]
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
    assert check_midis_equals(midi, midi_copy)[1]

    # Altering notes
    i = 0
    while i < len(midi_copy.tracks):
        if len(midi_copy.tracks[i].notes) > 0:
            midi_copy.tracks[i].notes[-1].pitch += 5
            assert not check_midis_equals(midi, midi_copy)[1]
            break
        i += 1

    # Altering track events
    if len(midi_copy.tracks) > 0:
        # Altering pedals
        midi_copy = copy(midi)
        if len(midi_copy.tracks[0].pedals) == 0:
            midi_copy.tracks[0].pedals.append(PedalTick(0, 10))
        else:
            midi_copy.tracks[0].pedals[-1].end += 10
        assert not check_midis_equals(midi, midi_copy)[1]

        # Altering pitch bends
        midi_copy = copy(midi)
        if len(midi_copy.tracks[0].pitch_bends) == 0:
            midi_copy.tracks[0].pitch_bends.append(PitchBend(50, 10))
        else:
            midi_copy.tracks[0].pitch_bends[-1].end += 10
        assert not check_midis_equals(midi, midi_copy)[1]

    # Altering tempos
    midi_copy = copy(midi)
    if len(midi_copy.tempos) == 0:
        midi_copy.tempos.append(Tempo(50, 10))
    else:
        midi_copy.tempos[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)[1]

    # Altering time signatures
    midi_copy = copy(midi)
    if len(midi_copy.time_signatures) == 0:
        midi_copy.time_signatures.append(TimeSignature(10, 4, 4))
    else:
        midi_copy.time_signatures[-1].time += 10
    assert not check_midis_equals(midi, midi_copy)[1]


def test_merge_tracks(
    midi_path: Union[str, Path] = MIDI_PATHS_ONE_TRACK[0],
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
def test_merge_same_program_tracks_and_by_class(midi_path: Union[str, Path]):
    midi = Score(midi_path)
    for track in midi.tracks:
        if track.is_drum:
            track.program = -1

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
        valid_programs=list(range(-1, 128)),
        filter_pitches=True,
    )


def test_nb_pos():
    (tok_conf := TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)).use_time_signatures = True
    tokenizer = REMI(tok_conf)
    midi = Score(MIDI_PATHS_ONE_TRACK[0])
    del_invalid_time_sig(midi.time_signatures, tokenizer.time_signatures)
    _ = nb_bar_pos(
        tokenizer.midi_to_tokens(midi)[0].ids,
        tokenizer["Bar_None"],
        tokenizer.token_ids_of_type("Position"),
    )
