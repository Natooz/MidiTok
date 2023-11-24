"""
Test validation methods.
# TODO make use of MidiFile __eq__?
"""

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from miditoolkit import (
    Instrument,
    Marker,
    MidiFile,
    Note,
    Pedal,
    PitchBend,
    TempoChange,
    TimeSignature,
)

import miditok
from miditok.constants import TIME_SIGNATURE_RANGE

SEED = 777
ALL_TOKENIZATIONS = miditok.tokenizations.__all__
TOKENIZATIONS_BPE = ["REMI", "MIDILike", "TSD", "MMM", "Structured"]

TIME_SIGNATURE_RANGE_TESTS = TIME_SIGNATURE_RANGE
TIME_SIGNATURE_RANGE_TESTS.update({2: [2, 3, 4]})
TIME_SIGNATURE_RANGE_TESTS[4].append(8)
# TODO centralize params?

HERE = Path(__file__).parent
MIDI_PATHS_ONE_TRACK = sorted((HERE / "MIDIs_one_track").rglob("*.mid"))
MIDI_PATHS_MULTITRACK = sorted((HERE / "MIDIs_multitrack").rglob("*.mid"))
MIDI_PATHS_ALL = sorted(
    deepcopy(MIDI_PATHS_ONE_TRACK) + deepcopy(MIDI_PATHS_MULTITRACK)
)
TEST_LOG_DIR = HERE / "test_logs"


def prepare_midi_for_tests(
    midi: MidiFile, sort_notes: bool = False, tokenizer: miditok.MIDITokenizer = None
) -> MidiFile:
    """Prepares a midi for test by returning a copy with tracks sorted, and optionally notes.
    It also

    :param midi: midi reference.
    :param sort_notes: whether to sort the notes. This is not necessary before tokenizing a MIDI, as the sorting
        will be performed by the tokenizer. (default: False)
    :param tokenizer: in order to downsample the MIDI before sorting its content.
    :return: a new MIDI object with track (and notes) sorted.
    """
    tokenization = type(tokenizer).__name__ if tokenizer is not None else None
    new_midi = deepcopy(midi)

    # Downsamples the MIDI if a tokenizer is given
    if tokenizer is not None:
        tokenizer.preprocess_midi(new_midi)

        # For Octuple/CPWord, as tempo is only carried at notes times, we need to adapt their times for comparison
        # Set tempo changes at onset times of notes
        # We use the first track only, as it is the one for which tempos are decoded
        if tokenizer.config.use_tempos and tokenization in ["Octuple", "CPWord"]:
            adapt_tempo_changes_times([new_midi.instruments[0]], new_midi.tempo_changes)

    for track in new_midi.instruments:
        # Adjust pedal ends to the maximum possible value
        if tokenizer is not None and tokenizer.config.use_sustain_pedals:
            adjust_pedal_durations(track.pedals, tokenizer, midi.ticks_per_beat)
        if track.is_drum:
            track.program = 0  # need to be done before sorting tracks per program
        if sort_notes:
            track.notes.sort(key=lambda x: (x.start, x.pitch, x.end, x.velocity))

    # Sorts tracks
    # MIDI detokenized with one_token_stream contains tracks sorted by note occurrence
    new_midi.instruments.sort(key=lambda x: (x.program, x.is_drum))

    return new_midi


def midis_equals(  # TODO replace this?
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


def notes_equals(note1: Note, note2: Note) -> str:  # TODO replace this
    if note1.start != note2.start:
        return "start"
    elif note1.end != note2.end:
        return "end"
    elif note1.pitch != note2.pitch:
        return "pitch"
    elif note1.velocity != note2.velocity:
        return "velocity"
    return ""


def tempo_changes_equals(  # TODO replace this
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


def time_signature_changes_equals(  # TODO replace this
    time_sig_changes1: List[TimeSignature], time_sig_changes2: List[TimeSignature]
) -> List[Tuple[str, Union[TimeSignature, int], float]]:
    if len(time_sig_changes1) != len(time_sig_changes2):
        return [("len", len(time_sig_changes1), len(time_sig_changes2))]
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


def pedal_equals(  # TODO replace this
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


def pitch_bend_equals(  # TODO replace this
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


def tokenize_and_check_equals(
    midi: MidiFile,
    tokenizer: miditok.MIDITokenizer,
    file_idx: Union[int, str],
    file_name: str,
) -> Tuple[MidiFile, bool]:
    has_errors = False
    tokenization = type(tokenizer).__name__
    midi.instruments.sort(key=lambda x: (x.program, x.is_drum))
    # merging is performed in preprocess only in one_token_stream mode
    # but in multi token stream, decoding will actually keep one track per program
    if tokenizer.config.use_programs:
        miditok.utils.merge_same_program_tracks(midi.instruments)

    tokens = tokenizer(midi)
    midi_decoded = tokenizer(
        tokens,
        miditok.utils.get_midi_programs(midi),
        time_division=midi.ticks_per_beat,
    )
    midi_decoded = prepare_midi_for_tests(
        midi_decoded, sort_notes=tokenization == "MIDILike"
    )

    # Checks types and values conformity following the rules
    err_tse = tokenizer.tokens_errors(tokens)
    if isinstance(err_tse, list):
        err_tse = sum(err_tse)
    if err_tse != 0.0:
        print(
            f"Validation of tokens types / values successions failed with {tokenization}: {err_tse:.2f}"
        )

    # Checks notes
    errors = midis_equals(midi, midi_decoded)
    if len(errors) > 0:
        has_errors = True
        for e, track_err in enumerate(errors):
            if track_err[-1][0][0] != "len":
                for err, note, exp in track_err[-1]:
                    midi_decoded.markers.append(
                        Marker(
                            f"{e}: with note {err} (pitch {note.pitch})",
                            note.start,
                        )
                    )
        print(
            f"MIDI {file_idx} - {file_name} / {tokenization} failed to encode/decode NOTES"
            f"({sum(len(t[2]) for t in errors)} errors)"
        )

    # Checks tempos
    if (
        tokenizer.config.use_tempos and tokenization != "MuMIDI"
    ):  # MuMIDI doesn't decode tempos
        tempo_errors = tempo_changes_equals(
            midi.tempo_changes, midi_decoded.tempo_changes
        )
        if len(tempo_errors) > 0:
            has_errors = True
            print(
                f"MIDI {file_idx} - {file_name} / {tokenization} failed to encode/decode TEMPO changes"
                f"({len(tempo_errors)} errors)"
            )

    # Checks time signatures
    if tokenizer.config.use_time_signatures:
        time_sig_errors = time_signature_changes_equals(
            midi.time_signature_changes,
            midi_decoded.time_signature_changes,
        )
        if len(time_sig_errors) > 0:
            has_errors = True
            print(
                f"MIDI {file_idx} - {file_name} / {tokenization} failed to encode/decode TIME SIGNATURE changes"
                f"({len(time_sig_errors)} errors)"
            )

    # Checks pedals
    if tokenizer.config.use_sustain_pedals:
        pedal_errors = pedal_equals(midi, midi_decoded)
        if any(len(err) > 0 for err in pedal_errors):
            has_errors = True
            print(
                f"MIDI {file_idx} - {file_name} / {tokenization} failed to encode/decode PEDALS"
                f"({sum(len(err) for err in pedal_errors)} errors)"
            )

    # Checks pitch bends
    if tokenizer.config.use_pitch_bends:
        pitch_bend_errors = pitch_bend_equals(midi, midi_decoded)
        if any(len(err) > 0 for err in pitch_bend_errors):
            has_errors = True
            print(
                f"MIDI {file_idx} - {file_name} / {tokenization} failed to encode/decode PITCH BENDS"
                f"({sum(len(err) for err in pitch_bend_errors)} errors)"
            )

    # TODO check control changes

    return midi_decoded, has_errors


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
    pedals: List[Pedal], tokenizer: miditok.MIDITokenizer, time_division: int
):
    """Adapt pedal offset times so that they match the possible durations covered by a tokenizer.

    :param pedals: list of Pedal objects to adapt.
    :param tokenizer: tokenizer (needed for durations).
    :param time_division: time division of the MIDI of origin.
    """
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
