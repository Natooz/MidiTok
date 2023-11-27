"""
Test validation methods.
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
    TempoChange,
)

import miditok
from miditok.constants import CHORD_MAPS, TIME_SIGNATURE_RANGE

SEED = 777

HERE = Path(__file__).parent
MIDI_PATHS_ONE_TRACK = sorted((HERE / "MIDIs_one_track").rglob("*.mid"))
MIDI_PATHS_MULTITRACK = sorted((HERE / "MIDIs_multitrack").rglob("*.mid"))
MIDI_PATHS_ALL = sorted(
    deepcopy(MIDI_PATHS_ONE_TRACK) + deepcopy(MIDI_PATHS_MULTITRACK)
)
TEST_LOG_DIR = HERE / "test_logs"

# TOKENIZATIONS
ALL_TOKENIZATIONS = miditok.tokenizations.__all__
TOKENIZATIONS_BPE = ["REMI", "MIDILike", "TSD", "MMM", "Structured"]

# TOK CONFIG PARAMS
TIME_SIGNATURE_RANGE_TESTS = TIME_SIGNATURE_RANGE
TIME_SIGNATURE_RANGE_TESTS.update({2: [2, 3, 4]})
TIME_SIGNATURE_RANGE_TESTS[4].append(8)
TOKENIZER_CONFIG_KWARGS = {
    "beat_res": {(0, 16): 8},  # TODO multiple res
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "num_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": TIME_SIGNATURE_RANGE_TESTS,
    "chord_maps": CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
}


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
        if (
            tokenizer.config.use_tempos
            and tokenization in ["Octuple", "CPWord"]
            and len(new_midi.instruments) > 0
        ):
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


def midis_notes_equals(
    midi1: MidiFile, midi2: MidiFile
) -> List[Tuple[int, str, List[Tuple[str, Union[Note, int], int]]]]:
    """Checks if the notes from two MIDIs are all equal, and if not returns the list of errors.

    :param midi1: first MIDI.
    :param midi2: second MIDI.
    :return: list of errors.
    """
    errors = []
    for track1, track2 in zip(midi1.instruments, midi2.instruments):
        track_errors = tracks_notes_equals(track1, track2)
        if len(track_errors) > 0:
            errors.append((track1.program, track1.name, track_errors))
    return errors


def tracks_notes_equals(
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


def check_midis_equals(
    midi1: MidiFile,
    midi2: MidiFile,
    check_tempos: bool = True,
    check_time_signatures: bool = True,
    check_pedals: bool = True,
    check_pitch_bends: bool = True,
    log_prefix: str = "",
) -> Tuple[MidiFile, bool]:
    has_errors = False
    types_of_errors = []

    # Checks notes and add markers if errors
    errors = midis_notes_equals(midi1, midi2)
    if len(errors) > 0:
        has_errors = True
        for e, track_err in enumerate(errors):
            if track_err[-1][0][0] != "len":
                for err, note, exp in track_err[-1]:
                    midi2.markers.append(
                        Marker(
                            f"{e}: with note {err} (pitch {note.pitch})",
                            note.start,
                        )
                    )
        print(
            f"{log_prefix} failed to encode/decode NOTES ({sum(len(t[2]) for t in errors)} errors)"
        )

    # Check pedals
    if check_pedals:
        for inst1, inst2 in zip(midi1.instruments, midi2.instruments):
            if inst1.pedals != inst2.pedals:
                types_of_errors.append("PEDALS")
                break

    # Check pitch bends
    if check_pitch_bends:
        for inst1, inst2 in zip(midi1.instruments, midi2.instruments):
            if inst1.pitch_bends != inst2.pitch_bends:
                types_of_errors.append("PITCH BENDS")
                break

    """# Check control changes
    if check_control_changes:
        for inst1, inst2 in zip(midi1.instruments, midi2.instruments):
            if inst1.control_changes != inst2.control_changes:
                types_of_errors.append("CONTROL CHANGES")
                break"""

    # Checks tempos
    if check_tempos:
        if midi1.tempo_changes != midi2.tempo_changes:
            types_of_errors.append("TEMPOS")

    # Checks time signatures
    if check_time_signatures:
        if midi1.time_signature_changes != midi2.time_signature_changes:
            types_of_errors.append("TIME SIGNATURES")

    # Prints types of errors
    has_errors = has_errors or len(types_of_errors) > 0
    for err_type in types_of_errors:
        print(f"{log_prefix} failed to encode/decode {err_type}")

    return midi2, not has_errors


def tokenize_and_check_equals(
    midi: MidiFile,
    tokenizer: miditok.MIDITokenizer,
    file_idx: Union[int, str],
    file_name: str,
) -> Tuple[MidiFile, bool]:
    tokenization = type(tokenizer).__name__
    log_prefix = f"MIDI {file_idx} - {file_name} / {tokenization}"
    midi.instruments.sort(key=lambda x: (x.program, x.is_drum))
    # merging is performed in preprocess only in one_token_stream mode
    # but in multi token stream, decoding will actually keep one track per program
    if tokenizer.config.use_programs:
        miditok.utils.merge_same_program_tracks(midi.instruments)

    # Tokenize and detokenize
    tokens = tokenizer(midi)
    midi_decoded = tokenizer(
        tokens,
        miditok.utils.get_midi_programs(midi),
        time_division=midi.ticks_per_beat,
    )
    midi_decoded = prepare_midi_for_tests(
        midi_decoded, sort_notes=tokenization == "MIDILike"
    )

    # Check decoded MIDI is identical
    midi_decoded, no_error = check_midis_equals(
        midi,
        midi_decoded,
        check_tempos=tokenizer.config.use_tempos and not tokenization == "MuMIDI",
        check_time_signatures=tokenizer.config.use_time_signatures,
        check_pedals=tokenizer.config.use_sustain_pedals,
        check_pitch_bends=tokenizer.config.use_pitch_bends,
        log_prefix=log_prefix,
    )

    # Checks types and values conformity following the rules
    err_tse = tokenizer.tokens_errors(tokens)
    if isinstance(err_tse, list):
        err_tse = sum(err_tse)
    if err_tse != 0.0:
        no_error = False
        print(f"{log_prefix} Validation of tokens types / values successions failed")

    return midi_decoded, not no_error


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
