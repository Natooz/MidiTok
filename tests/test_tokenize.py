"""Testing tokenization, making sure the data integrity is not altered."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from symusic import Score

import miditok
from miditok.constants import MIDI_LOADING_EXCEPTION

from .utils_tests import (
    ALL_TOKENIZATIONS,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
    MIDI_PATHS_ONE_TRACK_HARD,
    TEST_LOG_DIR,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
    tokenize_and_check_equals,
)

# Removing "hard" MIDIs from the list
MIDI_PATHS_ONE_TRACK = [
    p for p in MIDI_PATHS_ONE_TRACK if p not in MIDI_PATHS_ONE_TRACK_HARD
]

# One track params
default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_chords": False,  # set false to speed up tests
        "log_tempos": True,
        "chord_unknown": False,
        "delete_equal_successive_time_sig_changes": True,
        "delete_equal_successive_tempo_changes": True,
        "max_duration": (20, 0, 4),
    }
)
TOK_PARAMS_ONE_TRACK = []
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = deepcopy(default_params)
    params_.update(
        {
            "use_rests": True,
            "use_tempos": True,
            "use_time_signatures": True,
            "use_sustain_pedals": True,
            "use_pitch_bends": True,
            "use_pitch_intervals": True,
            "remove_duplicated_notes": True,
        }
    )
    adjust_tok_params_for_tests(tokenization_, params_)
    TOK_PARAMS_ONE_TRACK.append((tokenization_, params_))

_all_add_tokens = [
    "use_rests",
    "use_tempos",
    "use_time_signatures",
    "use_sustain_pedals",
    "use_pitch_bends",
    "use_pitch_intervals",
]
tokenizations_add_tokens = {
    "MIDILike": _all_add_tokens,
    "REMI": _all_add_tokens,
    "TSD": _all_add_tokens,
    "CPWord": ["use_rests", "use_tempos", "use_time_signatures"],
    "Octuple": ["use_tempos"],
    "MuMIDI": ["use_tempos"],
    "MMM": ["use_tempos", "use_time_signatures", "use_pitch_intervals"],
}
# Parametrize additional tokens
TOK_PARAMS_ONE_TRACK_HARD = []
for tokenization_ in ALL_TOKENIZATIONS:
    # If the tokenization isn't present we simply add one case with everything disabled
    add_tokens = tokenizations_add_tokens.get(tokenization_, _all_add_tokens[:1])
    if len(add_tokens) > 0:
        bin_combinations = [
            format(n, f"0{len(add_tokens)}b") for n in range(pow(2, len(add_tokens)))
        ]
    else:
        bin_combinations = ["0"]
    for bin_combination in bin_combinations:
        params_ = deepcopy(default_params)
        for param, bin_val in zip(add_tokens, bin_combination):
            params_[param] = bool(int(bin_val))
        TOK_PARAMS_ONE_TRACK_HARD.append((tokenization_, params_))

# Make final adjustments
for tpi in range(len(TOK_PARAMS_ONE_TRACK_HARD) - 1, -1, -1):
    # Delete cases for CPWord with rest and time signature
    tokenization_, params_ = TOK_PARAMS_ONE_TRACK_HARD[tpi]
    if (
        tokenization_ == "CPWord"
        and params_["use_rests"]
        and params_["use_time_signatures"]
    ):
        del TOK_PARAMS_ONE_TRACK_HARD[tpi]
        continue
    # Parametrize PedalOff cases for configurations using pedals
    if params_.get("use_sustain_pedals"):
        params_copy = deepcopy(params_)
        params_copy["sustain_pedal_duration"] = True
        TOK_PARAMS_ONE_TRACK_HARD.insert(tpi + 1, (tokenization_, params_copy))
for tokenization_, params_ in TOK_PARAMS_ONE_TRACK_HARD:
    adjust_tok_params_for_tests(tokenization_, params_)


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
) -> None:
    r"""
    Tokenize a MIDI file, decode it back and make sure it is identical to the ogi.

    The decoded MIDI should be identical to the original one after downsampling, and
    potentially notes deduplication.

    :param midi_path: path to the MIDI file to test.
    :param tok_params_set: tokenizer and its parameters to run.
    :param saving_erroneous_midis: will save MIDIs decoded with errors, to be used to
        debug.
    :param save_failed_midi_as_one_midi: will save the MIDI with conversion errors as a
        single MIDI, with all tracks appended altogether.
    """
    # Reads the MIDI and add pedal messages to make sure there are some
    try:
        midi = Score(Path(midi_path))
    except MIDI_LOADING_EXCEPTION as e:
        pytest.skip(f"Error when loading {midi_path.name}: {e}")

    # Creates the tokenizer
    tokenization, params = tok_params_set
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )
    str(tokenizer)  # shouldn't fail

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


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK)
def test_one_track_midi_to_tokens_to_midi(
    midi_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
    saving_erroneous_midis: bool = True,
):
    _test_tokenize(midi_path, tok_params_set, saving_erroneous_midis)


@pytest.mark.parametrize("midi_path", MIDI_PATHS_ONE_TRACK_HARD)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK_HARD)
def test_one_track_midi_to_tokens_to_midi_hard(
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
