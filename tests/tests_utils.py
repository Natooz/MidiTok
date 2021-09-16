""" Test validation methods

"""

from typing import Tuple, List, Union

from miditoolkit import MidiFile, Instrument, Note, TempoChange


def midis_equals(midi1: MidiFile, midi2: MidiFile) -> List[List[Tuple[str, Union[Note, int]]]]:
    errors = []
    for track1, track2 in zip(midi1.instruments, midi2.instruments):
        track_errors = track_equals(track1, track2)
        if len(track_errors) > 0:
            errors.append(track_errors)
    return errors


def track_equals(track1: Instrument, track2: Instrument) -> List[Tuple[str, Union[Note, int]]]:
    if len(track1.notes) != len(track2.notes):
        return [('len', 0)]
    errors = []
    for note1, note2 in zip(track1.notes, track2.notes):
        err = notes_equals(note1, note2)
        if err != '':
            errors.append((err, note2))
    return errors


def notes_equals(note1: Note, note2: Note) -> str:
    if note1.start != note2.start:
        return 'start'
    elif note1.end != note2.end:
        return 'end'
    elif note1.pitch != note2.pitch:
        return 'pitch'
    elif note1.velocity != note2.velocity:
        return 'velocity'
    return ''


def tempo_changes_equals(tempo_changes1: List[TempoChange], tempo_changes2: List[TempoChange]) \
        -> List[Tuple[str, float]]:
    errors = []
    for tempo_change1, tempo_change2 in zip(tempo_changes1, tempo_changes2):
        if tempo_change1.time != tempo_change2.time:
            errors.append(('time', tempo_change2.time))
        if tempo_change1.tempo != tempo_change2.tempo:
            errors.append(('tempo', tempo_change2.time))
    return errors
