#!/usr/bin/python3 python

"""
Testing the possible I/O formats of the tokenizers.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from symusic import Score

import miditok

from .utils import (
    ALL_TOKENIZATIONS,
    HERE,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
    tokenize_and_check_equals,
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


@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_IO)
def test_io_formats(
    tok_params_set: tuple[str, dict[str, Any]],
    midi_path: str | Path = HERE / "MIDIs_multitrack" / "Funkytown.mid",
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to
    MIDI files. The converted back MIDI files should identical to original one, expect
    with note starting and ending times quantized, and maybe a some duplicated notes
    removed.

    :param tok_params_set: tokenizer and its parameters to run.
    :param midi_path: path to the MIDI file to test.
    """
    midi = Score(midi_path)
    tokenization, params = tok_params_set
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )

    _, _, has_errors = tokenize_and_check_equals(midi, tokenizer, midi_path.stem)
    assert not has_errors
