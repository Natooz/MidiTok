#!/usr/bin/python3 python

"""One track test file
TODO rename file test_tokenize
"""

from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union

import pytest
from miditoolkit import MidiFile

import miditok
from miditok.constants import CHORD_MAPS

from .utils import (
    ALL_TOKENIZATIONS,
    MIDI_PATHS_ONE_TRACK,
    TEST_DIR,
    TIME_SIGNATURE_RANGE_TESTS,
    prepare_midi_for_tests,
    tokenize_and_check_equals,
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
    "use_pitch_intervals": True,
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "log_tempos": True,
    "time_signature_range": TIME_SIGNATURE_RANGE_TESTS,
    "chord_maps": CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": False,
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
    "sustain_pedal_duration": True,
    "one_token_stream_for_programs": False,
    "program_changes": True,
}


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
def test_one_track_midi_to_tokens_to_midi(
    midi_path: Union[str, Path],
    tokenizations: Sequence[str] = None,
    saving_erroneous_midis: bool = True,
):
    r"""Reads a MIDI file, converts it into tokens, convert it back to a MIDI object.
    The decoded MIDI should be identical to the original one after downsampling, and potentially notes deduplication.
    We only parametrize for midi files, as it would otherwise require to load them multiple times each.
    # TODO test parametrize tokenization / params_set

    :param tokenizations: sequence of tokenizer names to test.
    :param midi_path: path to the MIDI file to test.
    :param saving_erroneous_midis: will save MIDIs decoded with errors, to be used to debug.
    """
    if tokenizations is None:
        tokenizations = ALL_TOKENIZATIONS
    (out_path := TEST_DIR / "tokenization_errors").mkdir(exist_ok=True)
    at_least_one_error = False

    # Reads the midi
    midi = MidiFile(midi_path)
    # Will store the tracks tokenized / detokenized, to be saved in case of errors
    for ti, track in enumerate(midi.instruments):
        track.name = f"original {ti} not quantized"
    tracks_with_errors = []

    for tok_i, tokenization in enumerate(tokenizations):
        params = deepcopy(TOKENIZER_PARAMS)
        # Special beat res for test, up to 64 beats so the duration and time-shift values are
        # long enough for Structured, and with a single beat resolution
        if tokenization == "Structured":
            params["beat_res"] = {(0, 64): 8}
        elif tokenization == "Octuple":
            params["max_bar_embedding"] = 300
            params["use_time_signatures"] = False  # because of time shifted
        elif tokenization == "CPWord":
            # Rests and time sig can mess up with CPWord, when a Rest that is crossing new bar is followed
            # by a new TimeSig change, as TimeSig are carried with Bar tokens (and there is None is this case)
            if params["use_time_signatures"] and params["use_rests"]:
                params["use_rests"] = False

        tokenizer_config = miditok.TokenizerConfig(**params)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            tokenizer_config=tokenizer_config
        )

        # Process the MIDI
        # preprocess_midi is also performed when tokenizing, but we need to call it here for following adaptations
        midi_to_compare = prepare_midi_for_tests(midi, tokenizer=tokenizer)
        # Store preprocessed track
        if len(tracks_with_errors) == 0:
            tracks_with_errors += midi_to_compare.instruments
            for ti, track in enumerate(midi_to_compare.instruments):
                track.name = f"original {ti} quantized"

        # printing the tokenizer shouldn't fail
        _ = str(tokenizer)

        # MIDI -> Tokens -> MIDI
        decoded_midi, has_errors = tokenize_and_check_equals(
            midi_to_compare, tokenizer, tok_i, midi_path.stem
        )

        # Add track to error list
        if has_errors:
            for ti, track in enumerate(decoded_midi.instruments):
                track.name = f"{tok_i} encoded with {tokenization}"
            tracks_with_errors += decoded_midi.instruments

    # > 1 as the first one is the preprocessed
    if len(tracks_with_errors) > len(midi.instruments):
        at_least_one_error = True
        if saving_erroneous_midis:
            midi.tempo_changes = midi_to_compare.tempo_changes
            midi.time_signature_changes = midi_to_compare.time_signature_changes
            midi.instruments += tracks_with_errors
            midi.dump(out_path / midi_path.name)

    assert not at_least_one_error
