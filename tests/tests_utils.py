""" Test validation methods

"""

from typing import Tuple, List, Union

from miditoolkit import (
    MidiFile,
    Instrument,
    Note,
    TempoChange,
    TimeSignature,
    Pedal,
    PitchBend,
)
import numpy as np

from miditok import MIDITokenizer


ALL_TOKENIZATIONS = [
    "MIDILike",
    "TSD",
    "Structured",
    "REMI",
    "CPWord",
    "Octuple",
    "MuMIDI",
    "MMM",
]


def midis_equals(
    midi1: MidiFile, midi2: MidiFile
) -> List[Tuple[int, str, List[Tuple[str, Union[Note, int], int]]]]:
    errors = []
    for track1, track2 in zip(midi1.instruments, midi2.instruments):
        track_errors = track_equals(track1, track2)
        if len(track_errors) > 0:
            errors.append((track1.program, track1.name, track_errors))
    return errors


def track_equals(
    track1: Instrument, track2: Instrument
) -> List[Tuple[str, Union[Note, int], int]]:
    if len(track1.notes) != len(track2.notes):
        return [("len", len(track2.notes), len(track1.notes))]
    errors = []
    for note1, note2 in zip(track1.notes, track2.notes):
        err = notes_equals(note1, note2)
        if err != "":
            errors.append((err, note2, getattr(note1, err)))
    return errors


def notes_equals(note1: Note, note2: Note) -> str:
    if note1.start != note2.start:
        return "start"
    elif note1.end != note2.end:
        return "end"
    elif note1.pitch != note2.pitch:
        return "pitch"
    elif note1.velocity != note2.velocity:
        return "velocity"
    return ""


def tempo_changes_equals(
    tempo_changes1: List[TempoChange], tempo_changes2: List[TempoChange]
) -> List[Tuple[str, Union[TempoChange, int], float]]:
    if len(tempo_changes1) != len(tempo_changes2):
        return [("len", len(tempo_changes2), len(tempo_changes1))]
    errors = []
    for tempo_change1, tempo_change2 in zip(tempo_changes1, tempo_changes2):
        if tempo_change1.time != tempo_change2.time:
            errors.append(("time", tempo_change1, tempo_change2.time))
        if tempo_change1.tempo != tempo_change2.tempo:
            errors.append(("tempo", tempo_change1, tempo_change2.tempo))
    return errors


def time_signature_changes_equals(
    time_sig_changes1: List[TimeSignature], time_sig_changes2: List[TimeSignature]
) -> List[Tuple[str, TimeSignature, float]]:
    errors = []
    for time_sig_change1, time_sig_change2 in zip(time_sig_changes1, time_sig_changes2):
        if time_sig_change1.time != time_sig_change2.time:
            errors.append(("time", time_sig_change1, time_sig_change2.time))
        if time_sig_change1.numerator != time_sig_change2.numerator:
            errors.append(("numerator", time_sig_change1, time_sig_change2.numerator))
        if time_sig_change1.denominator != time_sig_change2.denominator:
            errors.append(
                ("denominator", time_sig_change1, time_sig_change2.denominator)
            )
    return errors


def pedal_equals(
    midi1: MidiFile, midi2: MidiFile
) -> List[List[Tuple[str, Union[Pedal, int], float]]]:
    errors = []
    for inst1, inst2 in zip(midi1.instruments, midi2.instruments):
        if len(inst1.pedals) != len(inst2.pedals):
            errors.append([("len", len(inst1.pedals), len(inst2.pedals))])
            continue
        errors.append([])
        for pedal1, pedal2 in zip(inst1.pedals, inst2.pedals):
            if pedal1.start != pedal2.start:
                errors[-1].append(("start", pedal1, pedal2.start))
            elif pedal1.end != pedal2.end:
                errors[-1].append(("end", pedal1, pedal2.end))
    return errors


def pitch_bend_equals(
    midi1: MidiFile, midi2: MidiFile
) -> List[List[Tuple[str, Union[PitchBend, int], float]]]:
    errors = []
    for inst1, inst2 in zip(midi1.instruments, midi2.instruments):
        if len(inst1.pitch_bends) != len(inst2.pitch_bends):
            errors.append([("len", len(inst1.pitch_bends), len(inst2.pitch_bends))])
            continue
        errors.append([])
        for pitch_bend1, pitch_bend2 in zip(inst1.pitch_bends, inst2.pitch_bends):
            if pitch_bend1.time != pitch_bend2.time:
                errors[-1].append(("time", pitch_bend1, pitch_bend2.time))
            elif pitch_bend1.pitch != pitch_bend2.pitch:
                errors[-1].append(("pitch", pitch_bend1, pitch_bend2.pitch))
    return errors


def adapt_tempo_changes_times(
    tracks: List[Instrument], tempo_changes: List[TempoChange]
):
    r"""Will adapt the times of tempo changes depending on the
    onset times of the notes of the MIDI.
    This is needed to pass the tempo tests for Octuple as the tempos
    will be decoded only from the notes.

    :param tracks: tracks of the MIDI to adapt the tempo changes
    :param tempo_changes: tempo changes to adapt
    """
    notes = sum((t.notes for t in tracks), [])
    notes.sort(key=lambda x: x.start)
    max_tick = max(note.start for note in notes)

    current_note_idx = 0
    tempo_idx = 1
    while tempo_idx < len(tempo_changes):
        if tempo_changes[tempo_idx].time > max_tick:
            del tempo_changes[tempo_idx]
            continue
        for n, note in enumerate(notes[current_note_idx:]):
            if note.start >= tempo_changes[tempo_idx].time:
                tempo_changes[tempo_idx].time = note.start
                current_note_idx += n
                break
        if tempo_changes[tempo_idx].time == tempo_changes[tempo_idx - 1].time:
            del tempo_changes[tempo_idx - 1]
            continue
        tempo_idx += 1


def adjust_pedal_durations(
    pedals: List[Pedal], tokenizer: MIDITokenizer, time_division: int
):
    durations_in_tick = np.array(
        [
            (beat * res + pos) * time_division // res
            for beat, pos, res in tokenizer.durations
        ]
    )
    for pedal in pedals:
        dur_index = np.argmin(np.abs(durations_in_tick - pedal.duration))
        beat, pos, res = tokenizer.durations[dur_index]
        dur_index_in_tick = (beat * res + pos) * time_division // res
        pedal.end = pedal.start + dur_index_in_tick
        pedal.duration = pedal.end - pedal.start


def remove_equal_successive_tempos(tempo_changes: List[TempoChange]):
    current_tempo = -1
    i = 0
    while i < len(tempo_changes):
        if tempo_changes[i].tempo == current_tempo:
            del tempo_changes[i]
            continue
        current_tempo = tempo_changes[i].tempo
        i += 1
