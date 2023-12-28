#!/usr/bin/python3 python

"""Multitrack test file
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from symusic import Pedal, Score

import miditok

from .utils import (
    ALL_TOKENIZATIONS,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
    TEST_LOG_DIR,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
    tokenize_and_check_equals,
)

# One track params
default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_chords": False,  # set false to speed up tests
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
        "max_duration": (20, 0, 4),
    }
)
TOK_PARAMS_ONE_TRACK = []
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = deepcopy(default_params)
    adjust_tok_params_for_tests(tokenization_, params_)
    TOK_PARAMS_ONE_TRACK.append((tokenization_, params_))


# Multitrack params
default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
# tempo decode fails without Rests for MIDILike because beat_res range is too short
default_params.update(
    {
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_sustain_pedals": True,
        "use_pitch_bends": True,
        "use_programs": True,
        "sustain_pedal_duration": False,
        "one_token_stream_for_programs": True,
        "program_changes": False,
    }
)
TOK_PARAMS_MULTITRACK = []
tokenizations_non_one_stream = [
    "TSD",
    "REMI",
    "MIDILike",
    "Structured",
    "CPWord",
    "Octuple",
]
tokenizations_program_change = ["TSD", "REMI", "MIDILike"]
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = deepcopy(default_params)
    adjust_tok_params_for_tests(tokenization_, params_)
    TOK_PARAMS_MULTITRACK.append((tokenization_, params_))

    if tokenization_ in tokenizations_non_one_stream:
        params_tmp = deepcopy(params_)
        params_tmp["one_token_stream_for_programs"] = False
        # Disable tempos for Octuple with one_token_stream_for_programs, as tempos are
        # carried by note tokens
        if tokenization_ == "Octuple":
            params_tmp["use_tempos"] = False
        TOK_PARAMS_MULTITRACK.append((tokenization_, params_tmp))
    if tokenization_ in tokenizations_program_change:
        params_tmp = deepcopy(params_)
        params_tmp["program_changes"] = True
        TOK_PARAMS_MULTITRACK.append((tokenization_, params_tmp))


def _test_tokenize(
    midi_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
    saving_erroneous_midis: bool = False,
    save_failed_midi_as_one_midi: bool = True,
):
    r"""Reads a MIDI file, converts it into tokens, convert it back to a MIDI object.
    The decoded MIDI should be identical to the original one after downsampling, and
    potentially notes deduplication. We only parametrize for midi files, as it would
    otherwise require to load them multiple times each.

    :param midi_path: path to the MIDI file to test.
    :param tok_params_set: tokenizer and its parameters to run.
    :param saving_erroneous_midis: will save MIDIs decoded with errors, to be used to
        debug.
    :param save_failed_midi_as_one_midi: will save the MIDI with conversion errors as a
        single MIDI, with all tracks appended altogether.
    """
    # Reads the MIDI and add pedal messages to make sure there are some
    midi = Score(Path(midi_path))

    # Creates the tokenizer
    tokenization, params = tok_params_set
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )
    str(tokenizer)  # shouldn't fail

    # Adds pedals to add a bit of complexity
    if tokenizer.config.use_sustain_pedals:
        for ti in range(min(3, len(midi.tracks))):
            midi.tracks[ti].pedals.extend(
                [Pedal(start, 200) for start in [100, 600, 1800, 2200]]
            )
            midi.tracks[ti].pedals.sort(key=lambda p: p.time)

    # MIDI -> Tokens -> MIDI
    midi_decoded, midi_ref, has_errors = tokenize_and_check_equals(
        midi, tokenizer, midi_path.stem
    )

    if has_errors and saving_erroneous_midis:
        TEST_LOG_DIR.mkdir(exist_ok=True, parents=True)
        if save_failed_midi_as_one_midi:
            for i in range(len(midi_decoded.tracks) - 1, -1, -1):
                midi_ref.tracks.insert(i + 1, midi_decoded.tracks[i])
            midi_ref.markers.extend(midi_decoded.markers)
        else:
            midi_decoded.dump_midi(
                TEST_LOG_DIR / f"{midi_path.stem}_{tokenization}_decoded.mid"
            )
        midi_ref.dump_midi(TEST_LOG_DIR / f"{midi_path.stem}_{tokenization}.mid")

    assert not has_errors


# TODO parametrize additional tokens
# @pytest.mark.parametrize("param", [False, True])
@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK)
def test_one_track_midi_to_tokens_to_midi(
    midi_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
    saving_erroneous_midis: bool = True,
):
    _test_tokenize(midi_path, tok_params_set, saving_erroneous_midis)


@pytest.mark.parametrize("midi_path", MIDI_PATHS_MULTITRACK)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_MULTITRACK)
def test_multitrack_midi_to_tokens_to_midi(
    midi_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
    saving_erroneous_midis: bool = False,
):
    _test_tokenize(midi_path, tok_params_set, saving_erroneous_midis)
