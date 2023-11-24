#!/usr/bin/python3 python

"""
Testing the possible I/O formats of the tokenizers.
"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import pytest
from miditoolkit import MidiFile

import miditok

from .utils import ALL_TOKENIZATIONS, HERE, prepare_midi_for_tests

BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "use_programs": False,
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "beat_res_rest": {(0, 16): 4},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": {4: [4]},
}

programs_tokenizations = ["TSD", "REMI", "MIDILike", "Structured", "CPWord"]
test_cases_programs = [
    (
        {
            "use_programs": True,
            "one_token_stream_for_programs": True,
            "program_changes": False,
        },
        [],
    ),
    (
        {
            "use_programs": True,
            "one_token_stream_for_programs": True,
            "program_changes": True,
        },
        ["Structured", "CPWord"],
    ),
    (
        {
            "use_programs": True,
            "one_token_stream_for_programs": False,
            "program_changes": False,
        },
        ["Structured"],
    ),
]


def encode_decode_and_check(tokenizer: miditok.MIDITokenizer, midi: MidiFile) -> bool:
    """Tests if a

    :param tokenizer:
    :param midi:
    :return:
    """
    # Process the MIDI
    midi_to_compare = prepare_midi_for_tests(midi)

    # Convert the midi to tokens, and keeps the ids (integers)
    tokens = tokenizer(midi_to_compare)
    if tokenizer.one_token_stream:
        tokens = tokens.ids
    else:
        tokens = [stream.ids for stream in tokens]

    # Convert back token ids to a MIDI object
    kwargs = {"time_division": midi.ticks_per_beat}
    if not tokenizer.one_token_stream:
        kwargs["programs"] = miditok.utils.get_midi_programs(midi_to_compare)
    try:
        decoded_midi = tokenizer(tokens, **kwargs)
    except Exception as e:
        print(f"Error when decoding token ids with {tokenizer.__class__.__name__}: {e}")
        return True

    # Checks its good
    decoded_midi = prepare_midi_for_tests(decoded_midi, sort_notes=True)
    return decoded_midi == midi_to_compare


@pytest.mark.parametrize("tokenization", ALL_TOKENIZATIONS)
def test_io_formats(
    tokenization: str,
    midi_path: Union[str, Path] = HERE / "MIDIs_multitrack" / "Funkytown.mid",
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed.

    :param tokenization: tokenization to test.
    """
    at_least_one_error = False
    midi = MidiFile(midi_path)
    # TODO test params centralized method?
    params = deepcopy(TOKENIZER_PARAMS)
    if tokenization == "Structured":
        params["beat_res"] = {(0, 512): 8}
    elif tokenization == "Octuple":
        params["use_time_signatures"] = False
    tokenizer_config = miditok.TokenizerConfig(**params)
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=tokenizer_config
    )

    at_least_one_error = encode_decode_and_check(tokenizer, midi) or at_least_one_error

    # If TSD, also test in use_programs / one_token_stream mode
    if tokenization in programs_tokenizations:
        for custom_params, excluded_tok in test_cases_programs:
            if tokenization in excluded_tok:
                continue
            params = deepcopy(TOKENIZER_PARAMS)
            params.update(custom_params)
            tokenizer_config = miditok.TokenizerConfig(**params)
            tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
                tokenizer_config=tokenizer_config
            )
            at_least_one_error = (
                encode_decode_and_check(tokenizer, midi) or at_least_one_error
            )

    assert not at_least_one_error
