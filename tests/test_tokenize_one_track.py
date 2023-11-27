#!/usr/bin/python3 python

"""One track test file
TODO rename file test_tokenize
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import pytest
from miditoolkit import MidiFile

import miditok

from .utils import (
    ALL_TOKENIZATIONS,
    MIDI_PATHS_ONE_TRACK,
    TEST_LOG_DIR,
    TOKENIZER_CONFIG_KWARGS,
    prepare_midi_for_tests,
    tokenize_and_check_equals,
)

default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_chords": False,  # set false to speed up tests as it takes some time on maestro MIDIs
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_sustain_pedals": True,
        "use_pitch_bends": True,
        "use_pitch_intervals": True,
        "log_tempos": True,
        "chord_unknown": False,
        "delete_equal_successive_time_sig_changes": True,
        "delete_equal_successive_tempo_changes": True,
        "sustain_pedal_duration": True,
    }
)
TOK_PARAMS_ONE_TRACK = []
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = deepcopy(default_params)
    if tokenization_ == "Structured":
        params_["beat_res"] = {(0, 64): 8}
    elif tokenization_ == "Octuple":
        params_["max_bar_embedding"] = 300
        params_["use_time_signatures"] = False  # because of time shifted
    elif tokenization_ == "CPWord":
        # Rests and time sig can mess up with CPWord, when a Rest that is crossing new bar is followed
        # by a new TimeSig change, as TimeSig are carried with Bar tokens (and there is None is this case)
        if params_["use_time_signatures"] and params_["use_rests"]:
            params_["use_rests"] = False
    TOK_PARAMS_ONE_TRACK.append((tokenization_, params_))


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
def test_one_track_midi_to_tokens_to_midi(
    midi_path: Union[str, Path],
    tok_params_sets: Sequence[Tuple[str, Dict[str, Any]]] = None,
    saving_erroneous_midis: bool = True,
):
    r"""Reads a MIDI file, converts it into tokens, convert it back to a MIDI object.
    The decoded MIDI should be identical to the original one after downsampling, and potentially notes deduplication.
    We only parametrize for midi files, as it would otherwise require to load them multiple times each.
    # TODO test parametrize tokenization / params_set, if faster --> unique method for test tok (one+multi)

    :param midi_path: path to the MIDI file to test.
    :param tok_params_sets: sequence of tokenizer and its parameters to run.
    :param saving_erroneous_midis: will save MIDIs decoded with errors, to be used to debug.
    """
    if tok_params_sets is None:
        tok_params_sets = TOK_PARAMS_ONE_TRACK
    at_least_one_error = False

    # Reads the midi
    midi = MidiFile(midi_path)
    # Will store the tracks tokenized / detokenized, to be saved in case of errors
    for ti, track in enumerate(midi.instruments):
        track.name = f"original {ti} not quantized"
    tracks_with_errors = []

    for tok_i, (tokenization, params) in enumerate(tok_params_sets):
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            tokenizer_config=miditok.TokenizerConfig(**params)
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
            TEST_LOG_DIR.mkdir(exist_ok=True, parents=True)
            midi.tempo_changes = midi_to_compare.tempo_changes
            midi.time_signature_changes = midi_to_compare.time_signature_changes
            midi.instruments += tracks_with_errors
            midi.dump(TEST_LOG_DIR / midi_path.name)

    assert not at_least_one_error
