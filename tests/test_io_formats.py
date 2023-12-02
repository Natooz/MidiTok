#!/usr/bin/python3 python

"""
Testing the possible I/O formats of the tokenizers.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pytest
from miditoolkit import MidiFile

import miditok

from .utils import (
    ALL_TOKENIZATIONS,
    HERE,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
    prepare_midi_for_tests,
)

default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_sustain_pedals": True,
        "use_pitch_bends": True,
    }
)
tokenizations_no_one_stream = [
    "TSD",
    "REMI",
    "MIDILike",
    "Structured",
    "CPWord",
    "Octuple",
]
configs = (
    {
        "use_programs": True,
        "one_token_stream_for_programs": True,
        "program_changes": False,
    },
    {
        "use_programs": True,
        "one_token_stream_for_programs": True,
        "program_changes": True,
    },
    {
        "use_programs": True,
        "one_token_stream_for_programs": False,
        "program_changes": False,
    },
)
TOK_PARAMS_IO = []
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = deepcopy(default_params)
    adjust_tok_params_for_tests(tokenization_, params_)
    TOK_PARAMS_IO.append((tokenization_, params_))

    if tokenization_ in tokenizations_no_one_stream:
        for config in configs:
            params_tmp = deepcopy(params_)
            params_tmp.update(config)
            TOK_PARAMS_IO.append((tokenization_, params_tmp))


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


@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_IO)
def test_io_formats(
    tok_params_set: Tuple[str, Dict[str, Any]],
    midi_path: Union[str, Path] = HERE / "MIDIs_multitrack" / "Funkytown.mid",
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to
    MIDI files. The converted back MIDI files should identical to original one, expect
    with note starting and ending times quantized, and maybe a some duplicated notes
    removed.

    :param tok_params_set: tokenizer and its parameters to run.
    :param midi_path: path to the MIDI file to test.
    """
    midi = MidiFile(midi_path)
    tokenization, params = tok_params_set
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )

    at_least_one_error = encode_decode_and_check(tokenizer, midi)
    assert not at_least_one_error
