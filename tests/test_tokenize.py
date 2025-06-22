"""Testing tokenization, making sure the data integrity is not altered."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from symusic import Score

import miditok
from miditok.constants import SCORE_LOADING_EXCEPTION, USE_NOTE_DURATION_PROGRAMS

from .utils_tests import (
    ABC_PATHS,
    ALL_TOKENIZATIONS,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
    MIDI_PATHS_ONE_TRACK_HARD,
    TEST_LOG_DIR,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
    tokenize_and_check_equals,
)

MMM_BASE_TOKENIZATIONS = ("TSD", "REMI", "MIDILike")

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
    if tokenization_ == "MMM":
        for tokenization__ in MMM_BASE_TOKENIZATIONS:
            params__ = params_.copy()
            params__["base_tokenizer"] = tokenization__
            adjust_tok_params_for_tests(tokenization_, params__)
            TOK_PARAMS_ONE_TRACK.append((tokenization_, params__))
    else:
        adjust_tok_params_for_tests(tokenization_, params_)
        TOK_PARAMS_ONE_TRACK.append((tokenization_, params_))

_all_add_tokens = [
    "use_velocities",
    "use_note_duration_programs",
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
    "CPWord": [
        "use_velocities",
        "use_note_duration_programs",
        "use_rests",
        "use_tempos",
        "use_time_signatures",
    ],
    "Octuple": ["use_velocities", "use_note_duration_programs", "use_tempos"],
    "MuMIDI": ["use_velocities", "use_note_duration_programs", "use_tempos"],
    "MMM": [
        "use_velocities",
        "use_note_duration_programs",
        "use_tempos",
        "use_time_signatures",
        "use_pitch_intervals",
    ],
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
            bool_val = bool(int(bin_val))
            if param == "use_note_duration_programs":
                params_[param] = USE_NOTE_DURATION_PROGRAMS if bool_val else []
            else:
                params_[param] = bool_val
        if tokenization_ == "MMM":
            for tokenization__ in MMM_BASE_TOKENIZATIONS:
                params__ = params_.copy()
                params__["base_tokenizer"] = tokenization__
                TOK_PARAMS_ONE_TRACK_HARD.append((tokenization_, params__))
        else:
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
    if tokenization_ == "MMM":
        for tokenization__ in MMM_BASE_TOKENIZATIONS:
            params__ = params_.copy()
            params__["base_tokenizer"] = tokenization__
            adjust_tok_params_for_tests(tokenization_, params__)
            TOK_PARAMS_MULTITRACK.append((tokenization_, params__))
    else:
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
    file_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
    saving_erroneous_files: bool = False,
    save_failed_file_as_one_file: bool = True,
) -> None:
    r"""
    Tokenize a music file, decode it back and make sure it is identical to the ogi.

    The decoded music score should be identical to the original one after downsampling,
    and potentially notes deduplication.
    s
    :param file_path: path to the music file to test.
    :param tok_params_set: tokenizer and its parameters to run.
    :param saving_erroneous_files: will save music scores decoded with errors, to be
        used to debug.
    :param save_failed_file_as_one_file: will save the music scores with conversion
        errors as a single file, with all tracks appended altogether.
    """
    # Reads the music file and add pedal messages to make sure there are some
    try:
        score = Score(Path(file_path))
    except SCORE_LOADING_EXCEPTION as e:
        pytest.skip(f"Error when loading {file_path.name}: {e}")

    # Creates the tokenizer
    tokenization, params = tok_params_set
    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )
    str(tokenizer)  # shouldn't fail

    # Score -> Tokens -> Score
    score_decoded, score_ref, has_errors = tokenize_and_check_equals(
        score, tokenizer, file_path.stem
    )

    if has_errors and saving_erroneous_files:
        TEST_LOG_DIR.mkdir(exist_ok=True, parents=True)
        if save_failed_file_as_one_file:
            for i in range(len(score_decoded.tracks) - 1, -1, -1):
                score_ref.tracks.insert(i + 1, score_decoded.tracks[i])
            score_ref.markers.extend(score_decoded.markers)
        else:
            score_decoded.dump_midi(
                TEST_LOG_DIR / f"{file_path.stem}_{tokenization}_decoded.mid"
            )
        score_ref.dump_midi(TEST_LOG_DIR / f"{file_path.stem}_{tokenization}.mid")

    assert not has_errors


def _id_tok(tok_params_set: tuple[str, dict]) -> str:
    """
    Return the "id" of a tokenizer params set.

    :param tok_params_set: tokenizer params set.
    :return: id
    """
    return tok_params_set[0]


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK, ids=_id_tok)
def test_one_track_midi_to_tokens_to_midi(
    file_path: str | Path, tok_params_set: tuple[str, dict[str, Any]]
):
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=True)


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK_HARD, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK_HARD, ids=_id_tok)
def test_one_track_midi_to_tokens_to_midi_hard(
    file_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
):
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=True)


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_MULTITRACK, ids=_id_tok)
def test_multitrack_midi_to_tokens_to_midi(
    file_path: str | Path, tok_params_set: tuple[str, dict[str, Any]]
):
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=False)


@pytest.mark.parametrize("file_path", ABC_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK, ids=_id_tok)
def test_abc_to_tokens_to_abc(
    file_path: str | Path, tok_params_set: tuple[str, dict[str, Any]]
):
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=False)
