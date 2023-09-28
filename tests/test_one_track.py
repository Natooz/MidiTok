#!/usr/bin/python3 python

"""One track test file
"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
from time import time

import miditok
from miditok.constants import TIME_SIGNATURE_RANGE, CHORD_MAPS
from miditoolkit import MidiFile
from tqdm import tqdm

from .tests_utils import (
    ALL_TOKENIZATIONS,
    tokenize_check_equals,
    adapt_tempo_changes_times,
    adjust_pedal_durations,
    remove_equal_successive_tempos,
)

TIME_SIGNATURE_RANGE.update({2: [2, 3, 4]})
BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": False,  # set false to speed up tests as it takes some time on maestro MIDIs
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "use_programs": True,
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "log_tempos": True,
    "time_signature_range": TIME_SIGNATURE_RANGE,
    "chord_maps": CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": False,
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
    "sustain_pedal_duration": True,
    "one_token_stream_for_programs": False,
    "program_changes": True,
}


def test_one_track_midi_to_tokens_to_midi(
    data_path: Union[str, Path, PurePath] = "./tests/One_track_MIDIs",
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
            params = deepcopy(TOKENIZER_PARAMS)
            # Special beat res for test, up to 64 beats so the duration and time-shift values are
            # long enough for Structured, and with a single beat resolution
            if tokenization == "Structured":
                params["beat_res"] = {(0, 64): 8}
            elif tokenization == "Octuple":
                params["max_bar_embedding"] = 300
                params["use_time_signatures"] = False  # because of time shifted

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
            # Same for CPWord which carries tempo with Position (for notes)
            if tokenization in ["Octuple", "CPWord"]:
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

            # MIDI -> Tokens -> MIDI
            decoded_midi, has_errors = tokenize_check_equals(
                midi_to_compare, tokenizer, i, file_path.stem
            )

            # Add track to error list
            if has_errors:
                tracks_with_errors.append(decoded_midi.instruments[0])
                tracks_with_errors[-1].name = f"encoded with {tokenization}"

        # > 1 as the first one is the preprocessed
        if len(tracks_with_errors) > 1:
            at_least_one_error = True
            if saving_erroneous_midis:
                midi.tempo_changes = midi_to_compare.tempo_changes
                midi.time_signature_changes = midi_to_compare.time_signature_changes
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
        default="tests/One_track_MIDIs",
        help="directory of MIDI files to use for test",
    )
    args = parser.parse_args()
    test_one_track_midi_to_tokens_to_midi(args.data)
