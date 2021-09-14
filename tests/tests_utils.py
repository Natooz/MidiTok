""" Test validation methods

"""

from typing import Tuple, List, Union

from miditoolkit import Instrument, Note, TempoChange


def track_equals(track1: Instrument, track2: Instrument) -> Tuple[bool, List[Tuple[str, Union[Note, int]]]]:
    if len(track1.notes) != len(track2.notes):
        return False, [('len', 0)]
    errors = []
    for note1, note2 in zip(track1.notes, track2.notes):
        equals, err = notes_equals(note1, note2)
        if not equals:
            errors.append((err, note2))
    return False if len(errors) > 0 else True, errors


def notes_equals(note1: Note, note2: Note) -> Tuple[bool, str]:
    if note1.start != note2.start:
        return False, 'start'
    elif note1.end != note2.end:
        return False, 'end'
    elif note1.pitch != note2.pitch:
        return False, 'pitch'
    elif note1.velocity != note2.velocity:
        return False, 'velocity'
    return True, ''


def tempo_changes_equals(tempo_changes1: List[TempoChange], tempo_changes2: List[TempoChange]) \
        -> Tuple[bool, List[Tuple[str, float]]]:
    errors = []
    for tempo_change1, tempo_change2 in zip(tempo_changes1, tempo_changes2):
        if tempo_change1.time != tempo_change2.time:
            errors.append(('time', tempo_change2.time))
        if tempo_change1.tempo != tempo_change2.tempo:
            errors.append(('tempo', tempo_change2.time))
    return False if len(errors) > 0 else True, errors
