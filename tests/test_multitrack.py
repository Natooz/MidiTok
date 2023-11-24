#!/usr/bin/python3 python

"""Multitrack test file
"""

from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union

import pytest
from miditoolkit import MidiFile, Pedal

import miditok

from .utils import (
    ALL_TOKENIZATIONS,
    MIDI_PATHS_MULTITRACK,
    TEST_LOG_DIR,
    prepare_midi_for_tests,
    tokenize_and_check_equals,
)

BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": True,
    "use_rests": True,  # tempo decode fails when False for MIDILike because beat_res range is too short
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "use_programs": True,
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "log_tempos": False,
    "sustain_pedal_duration": False,
    "one_token_stream_for_programs": True,
    "program_changes": False,
}

# Define kwargs sets
# The first set is empty, using the default params
params_kwargs_sets = {tok: [{}] for tok in ALL_TOKENIZATIONS}
programs_tokenizations = ["TSD", "REMI", "MIDILike", "Structured", "CPWord", "Octuple"]
for tok in programs_tokenizations:
    params_kwargs_sets[tok].append(
        {"one_token_stream_for_programs": False},
    )
for tok in ["TSD", "REMI", "MIDILike"]:
    params_kwargs_sets[tok].append(
        {"program_changes": True},
    )
# Disable tempos for Octuple with one_token_stream_for_programs, as tempos are carried by note tokens, and
# time signatures for the same reasons (as time could be shifted by on or several bars)
params_kwargs_sets["Octuple"][1]["use_tempos"] = False
params_kwargs_sets["Octuple"][0]["use_time_signatures"] = False
params_kwargs_sets["Octuple"][1]["use_time_signatures"] = False
# Increase the TimeShift voc for Structured as it doesn't support successive TimeShifts
for kwargs_set in params_kwargs_sets["Structured"]:
    kwargs_set["beat_res"] = {(0, 512): 8}


@pytest.mark.parametrize("midi_path", MIDI_PATHS_MULTITRACK)
def test_multitrack_midi_to_tokens_to_midi(
    midi_path: Union[str, Path],
    tokenizations: Sequence[str] = None,
    saving_erroneous_midis: bool = False,
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
    at_least_one_error = False

    # Reads the MIDI and add pedal messages
    midi = MidiFile(Path(midi_path))
    for ti in range(max(3, len(midi.instruments))):
        midi.instruments[ti].pedals = [
            Pedal(start, start + 200) for start in [100, 600, 1800, 2200]
        ]

    for tok_i, tokenization in enumerate(tokenizations):
        for pi, params_kwargs in enumerate(params_kwargs_sets[tokenization]):
            idx = f"{tok_i}_{pi}"
            params = deepcopy(TOKENIZER_PARAMS)
            params.update(params_kwargs)
            tokenizer_config = miditok.TokenizerConfig(**params)
            tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
                tokenizer_config=tokenizer_config
            )

            # Process the MIDI
            # midi notes / tempos / time signature quantized with the line above
            midi_to_compare = prepare_midi_for_tests(midi, tokenizer=tokenizer)

            # MIDI -> Tokens -> MIDI
            decoded_midi, has_errors = tokenize_and_check_equals(
                midi_to_compare, tokenizer, idx, midi_path.stem
            )

            if has_errors:
                TEST_LOG_DIR.mkdir(exist_ok=True, parents=True)
                at_least_one_error = True
                if saving_erroneous_midis:
                    decoded_midi.dump(
                        TEST_LOG_DIR / f"{midi_path.stem}_{tokenization}.mid"
                    )
                    midi_to_compare.dump(
                        TEST_LOG_DIR / f"{midi_path.stem}_{tokenization}_original.mid"
                    )

    assert not at_least_one_error
