#!/usr/bin/python3 python

"""One track test file
"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
from time import time

import miditok
from miditoolkit import MidiFile, Marker
from tqdm import tqdm

from .tests_utils import (
    ALL_TOKENIZATIONS,
    track_equals,
    tempo_changes_equals,
    pedal_equals,
    pitch_bend_equals,
    time_signature_changes_equals,
    adapt_tempo_changes_times,
    adjust_pedal_durations,
    remove_equal_successive_tempos,
)

BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": False,  # set false to speed up tests as it takes some time on maestro MIDIs
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "use_programs": False,
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "log_tempos": True,
    "time_signature_range": {4: [4]},
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": False,
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
    "sustain_pedal_duration": True,
}


def test_one_track_midi_to_tokens_to_midi(
    data_path: Union[str, Path, PurePath] = "./tests/Maestro_MIDIs",
    saving_erroneous_midis: bool = True,
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    :param data_path: root path to the data to test
    :param saving_erroneous_midis: will save MIDIs converted back with errors, to be used to debug
    """
    files = list(Path(data_path).glob("**/*.mid"))
    at_least_one_error = False
    t0 = time()

    for i, file_path in enumerate(tqdm(files, desc="Testing One Track")):
        # Reads the midi
        midi = MidiFile(file_path)
        # Will store the tracks tokenized / detokenized, to be saved in case of errors
        midi.instruments[0].name = "original not quantized"
        tracks_with_errors = []

        for tokenization in ALL_TOKENIZATIONS:
            has_errors = False
            params = deepcopy(TOKENIZER_PARAMS)
            # Special beat res for test, up to 64 beats so the duration and time-shift values are
            # long enough for Structured, and with a single beat resolution
            if tokenization == "Structured":
                params["beat_res"] = {(0, 64): 8}

            tokenizer_config = miditok.TokenizerConfig(**params)
            tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
                tokenizer_config=tokenizer_config
            )

            # Process the MIDI
            # midi notes / tempos / time signature quantized with the line above
            midi_to_compare = deepcopy(midi)
            for track in midi_to_compare.instruments:
                if track.is_drum:
                    track.program = (
                        0  # need to be done before sorting tracks per program
                    )

            # This step is also performed in preprocess_midi, but we need to call it here for the assertions below
            tokenizer.preprocess_midi(midi_to_compare)
            # For Octuple, as tempo is only carried at notes times, we need to adapt their times for comparison
            if tokenization in ["Octuple"]:
                adapt_tempo_changes_times(
                    midi_to_compare.instruments, midi_to_compare.tempo_changes
                )
            # When the tokenizer only decoded tempo changes different from the last tempo val
            if tokenization in ["CPWord"]:
                remove_equal_successive_tempos(midi_to_compare.tempo_changes)
            # Adjust pedal ends to the maximum possible value
            if tokenizer.config.use_sustain_pedals:
                for track in midi_to_compare.instruments:
                    adjust_pedal_durations(track.pedals, tokenizer, midi.ticks_per_beat)
            # Store preprocessed track
            if len(tracks_with_errors) == 0:
                tracks_with_errors.append(midi_to_compare.instruments[0])

            # printing the tokenizer shouldn't fail
            _ = str(tokenizer)

            # Convert the track in tokens
            tokens = tokenizer(midi_to_compare)
            if not tokenizer.one_token_stream:
                tokens = tokens[0]

            # Checks types and values conformity following the rules
            tokens_types = tokenizer.tokens_errors(tokens)
            if tokens_types != 0.0:
                print(
                    f"Validation of tokens types / values successions failed with {tokenization}: {tokens_types:.2f}"
                )

            # Convert back tokens into a track object
            if not tokenizer.one_token_stream:
                tokens = [tokens]
            new_midi = tokenizer.tokens_to_midi(
                tokens, time_division=midi_to_compare.ticks_per_beat
            )
            track = new_midi.instruments[0]
            track.name = f"encoded with {tokenization}"

            # Checks its good
            errors = track_equals(midi_to_compare.instruments[0], track)
            if len(errors) > 0:
                has_errors = True
                if errors[0][0] != "len":
                    for err, note, exp in errors:
                        midi.markers.append(
                            Marker(
                                f"ERR {tokenization} with note {err} (pitch {note.pitch})",
                                note.start,
                            )
                        )
                print(
                    f"MIDI {i} - {file_path} failed to encode/decode NOTES with {tokenization} ({len(errors)} errors)"
                )

            # Checks tempos
            if tokenizer.config.use_tempos and tokenization != "MuMIDI":
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
            if tokenizer.config.use_time_signatures:
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

            # Checks pedals
            if tokenizer.config.use_sustain_pedals:
                pedal_errors = pedal_equals(midi_to_compare, new_midi)
                if any(len(err) > 0 for err in pedal_errors):
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path} failed to encode/decode PEDALS with "
                        f"{tokenization} ({sum(len(err) for err in pedal_errors)} errors)"
                    )

            # Checks pitch bends
            if tokenizer.config.use_pitch_bends:
                pitch_bend_errors = pitch_bend_equals(midi_to_compare, new_midi)
                if any(len(err) > 0 for err in pitch_bend_errors):
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path} failed to encode/decode PITCH BENDS with "
                        f"{tokenization} ({sum(len(err) for err in pitch_bend_errors)} errors)"
                    )

            # TODO check control changes

            # Add track to error list
            if has_errors:
                tracks_with_errors.append(new_midi.instruments[0])
                tracks_with_errors[-1].name = f"encoded with {tokenization}"

        # > 1 as the first one is the preprocessed
        if len(tracks_with_errors) > 1:
            at_least_one_error = True
            if saving_erroneous_midis:
                midi.instruments += tracks_with_errors
                midi.instruments[1].name = "original quantized"
                midi.dump(PurePath("tests", "test_results", file_path.name))

    ttotal = time() - t0
    print(f"Took {ttotal:.2f} seconds")
    assert not at_least_one_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Encoding test")
    parser.add_argument(
        "--data",
        type=str,
        default="tests/Maestro_MIDIs",
        help="directory of MIDI files to use for test",
    )
    args = parser.parse_args()
    test_one_track_midi_to_tokens_to_midi(args.data)
