#!/usr/bin/python3 python

""" Multitrack test file
This test method will encode every track of a MIDI file.
These file contains tracks with long empty sections with no notes. Hence encodings
in which time is based on time-shift tokens (MIDI-like, Structured) will probably
not be suited for these files.

Structured and MIDI-Like are then not tested here.
You can still manage to make them work and pass the test be using a vocabulary
with large duration / time-shift values, but this is probably not suited for real
case situations.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import miditok
from miditoolkit import MidiFile
from tqdm import tqdm

from .tests_utils import (
    midis_equals,
    tempo_changes_equals,
    reduce_note_durations,
    adapt_tempo_changes_times,
    time_signature_changes_equals,
)

# Special beat res for test, up to 16 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 16): 8}
ADDITIONAL_TOKENS_TEST = {
    "Chord": True,
    "Rest": True,
    "Tempo": True,
    "TimeSignature": True,
    "Program": True,
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "rest_range": (
        4,
        1024,
    ),  # very high value to cover every possible rest in the test files
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": (16, 2),
}


def test_multitrack_midi_to_tokens_to_midi(
    data_path: Union[str, Path] = "./tests/Multitrack_MIDIs",
    saving_erroneous_midis: bool = False,
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    """
    tokenizations = ["REMI", "REMIPlus", "CPWord", "Octuple", "OctupleMono", "MuMIDI"]
    files = list(Path(data_path).glob("**/*.mid"))
    at_least_one_error = False

    for i, file_path in enumerate(tqdm(files, desc="Testing multitrack")):
        # Reads the MIDI
        try:
            midi = MidiFile(Path(file_path))
        except (
            Exception
        ):  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            continue
        if midi.ticks_per_beat % max(BEAT_RES_TEST.values()) != 0:
            continue
        has_errors = False

        for tokenization in tokenizations:
            tokenizer = getattr(miditok, tokenization)(
                beat_res=BEAT_RES_TEST,
                additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST),
            )

            # Process the MIDI
            midi_to_compare = deepcopy(
                midi
            )  # midi notes / tempos / time signature quantized with the line above
            for track in midi_to_compare.instruments:
                if track.is_drum:
                    track.program = (
                        0  # need to be done before sorting tracks per program
                    )

            # Sort and merge tracks if needed
            # MIDI produced with Octuple contains tracks ordered by program
            if tokenization in ["Octuple", "MuMIDI", "REMIPlus"]:
                miditok.utils.merge_same_program_tracks(
                    midi_to_compare.instruments
                )  # merge tracks
            for (
                track
            ) in midi_to_compare.instruments:  # reduce the duration of notes to long
                reduce_note_durations(
                    track.notes,
                    max(tu[1] for tu in BEAT_RES_TEST) * midi_to_compare.ticks_per_beat,
                )
                miditok.utils.remove_duplicated_notes(track.notes)
            if tokenization in ["Octuple", "REMIPlus"]:  # needed
                adapt_tempo_changes_times(
                    midi_to_compare.instruments, midi_to_compare.tempo_changes
                )

            # MIDI -> Tokens -> MIDI
            midi_to_compare.instruments.sort(
                key=lambda x: (x.program, x.is_drum)
            )  # sort tracks
            tokens = tokenizer(midi_to_compare)
            new_midi = tokenizer(
                tokens,
                miditok.utils.get_midi_programs(midi_to_compare),
                time_division=midi_to_compare.ticks_per_beat,
            )
            new_midi.instruments.sort(key=lambda x: (x.program, x.is_drum))

            # Checks types and values conformity following the rules
            tokens_types = tokenizer.tokens_errors(
                tokens[0] if not tokenizer.unique_track else tokens
            )
            if tokens_types != 0.0:
                print(
                    f"Validation of tokens types / values successions failed with {tokenization}: {tokens_types:.2f}"
                )

            # Checks notes
            errors = midis_equals(midi_to_compare, new_midi)
            if len(errors) > 0:
                has_errors = True
                """for track_err in errors:
                    if track_err[-1][0] != 'len':
                        for err, note, exp in track_err[-1]:
                            new_midi.markers.append(Marker(f'ERR {tokenization} with note {err} (pitch {note.pitch})',
                                                           note.start))"""
                print(
                    f"MIDI {i} - {file_path} failed to encode/decode NOTES with "
                    f"{tokenization} ({sum(len(t[2]) for t in errors)} errors)"
                )
                # return False

            # Checks tempos
            if (
                tokenizer.additional_tokens["Tempo"] and tokenization != "MuMIDI"
            ):  # MuMIDI doesn't decode tempos
                tempo_errors = tempo_changes_equals(
                    midi_to_compare.tempo_changes, new_midi.tempo_changes
                )
                if len(tempo_errors) > 0:
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path} failed to encode/decode TEMPO changes with "
                        f"{tokenization} ({len(tempo_errors)} errors)"
                    )

            # Checks time signatures
            if tokenizer.additional_tokens["TimeSignature"] and tokenization in [
                "Octuple",
                "REMIPlus",
            ]:
                time_sig_errors = time_signature_changes_equals(
                    midi_to_compare.time_signature_changes,
                    new_midi.time_signature_changes,
                )
                if len(time_sig_errors) > 0:
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path} failed to encode/decode TIME SIGNATURE changes with "
                        f"{tokenization} ({len(time_sig_errors)} errors)"
                    )

            if has_errors:
                at_least_one_error = True
                if saving_erroneous_midis:
                    new_midi.dump(
                        Path(
                            "tests",
                            "test_results",
                            f"{file_path.stem}_{tokenization}.mid",
                        )
                    )
                    midi_to_compare.dump(
                        Path(
                            "tests",
                            "test_results",
                            f"{file_path.stem}_{tokenization}_original.mid",
                        )
                    )
    assert not at_least_one_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Encoding test")
    parser.add_argument(
        "--data",
        type=str,
        default="tests/Multitrack_MIDIs",
        help="directory of MIDI files to use for test",
    )
    args = parser.parse_args()

    test_multitrack_midi_to_tokens_to_midi(args.data)
