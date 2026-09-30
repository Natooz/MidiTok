"""Testing tokenization, making sure the data integrity is not altered."""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

import miditok
from miditok.constants import (
    DEFAULT_TOKENIZER_FILE_NAME,
    SCORE_LOADING_EXCEPTION,
    USE_NOTE_DURATION_PROGRAMS,
)

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
    "use_control_changes",
    "use_pitch_intervals",
]
tokenizations_add_tokens = {
    "MIDILike": _all_add_tokens,
    "REMI": _all_add_tokens,
    "TSD": _all_add_tokens,
    "BEAT": [
        "use_velocities",
        "use_tempos",
        "use_key_signatures",
    ],
    "PerTok": ["use_velocities", "use_control_changes"],
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
        for param, bin_val in zip(add_tokens, bin_combination, strict=False):
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
    # Delete cases using rests without note durations
    if TOK_PARAMS_ONE_TRACK_HARD[tpi][1].get("use_rests") and not (
        TOK_PARAMS_ONE_TRACK_HARD[tpi][1].get("use_note_duration_programs")
    ):
        del TOK_PARAMS_ONE_TRACK_HARD[tpi]
        continue

    # Delete cases for CPWord with rest and time signature
    tokenization_, params_ = TOK_PARAMS_ONE_TRACK_HARD[tpi]
    if (
        tokenization_ == "CPWord"
        and params_["use_rests"]
        and params_["use_time_signatures"]
    ):
        del TOK_PARAMS_ONE_TRACK_HARD[tpi]
        continue
    # CC64 disables legacy pedals; precedence is covered by its dedicated test.
    if params_.get("use_control_changes") and params_.get("use_sustain_pedals"):
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
) -> None:
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=True)


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK_HARD, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK_HARD, ids=_id_tok)
def test_one_track_midi_to_tokens_to_midi_hard(
    file_path: str | Path,
    tok_params_set: tuple[str, dict[str, Any]],
) -> None:
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=True)


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_MULTITRACK, ids=_id_tok)
def test_multitrack_midi_to_tokens_to_midi(
    file_path: str | Path, tok_params_set: tuple[str, dict[str, Any]]
) -> None:
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=False)


@pytest.mark.parametrize("file_path", ABC_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_ONE_TRACK, ids=_id_tok)
def test_abc_to_tokens_to_abc(
    file_path: str | Path, tok_params_set: tuple[str, dict[str, Any]]
) -> None:
    _test_tokenize(file_path, tok_params_set, saving_erroneous_files=False)


@pytest.mark.filterwarnings(
    "ignore:Key signatures are not supported by "
    "(CPWord|Octuple|MuMIDI|Structured),:UserWarning"
)
def test_key_signatures_round_trip() -> None:
    """Test that key signatures are encoded and decoded back identically."""
    score = Score(480)
    score.tracks.append(
        Track(
            program=0,
            is_drum=False,
            notes=[Note(0, 240, 60, 100), Note(240, 240, 64, 100)],
        )
    )
    score.tracks.append(
        Track(program=24, is_drum=False, notes=[Note(120, 240, 67, 100)])
    )
    score.key_signatures.append(KeySignature(0, 3, 1))

    tokenizations_params = {
        "BEAT": {},
        "REMI": {},
        "TSD": {},
        "MIDILike": {},
        "PerTok": {
            "beat_res": {(0, 128): 4, (0, 32): 3},
            "use_microtiming": True,
            "ticks_per_quarter": 220,
            "max_microtiming_shift": 0.25,
            "num_microtiming_bins": 110,
        },
    }
    for tokenization, params in tokenizations_params.items():
        config = miditok.TokenizerConfig(use_key_signatures=True, **params)
        tokenizer = getattr(miditok, tokenization)(tokenizer_config=config)
        assert tokenizer.config.use_key_signatures

        tokens = tokenizer(score)
        decoded = tokenizer(tokens)
        assert list(decoded.key_signatures) == [KeySignature(0, 3, 1)]

    # MMM duplicates global tokens per track and decodes them from the first one only.
    for base_tokenization in MMM_BASE_TOKENIZATIONS:
        config = miditok.TokenizerConfig(
            use_key_signatures=True, base_tokenizer=base_tokenization
        )
        tokenizer = miditok.MMM(tokenizer_config=config)
        assert tokenizer.config.use_key_signatures

        tokens = tokenizer(score)
        assert tokens.tokens.count("KeySig_3:1") == len(score.tracks)
        decoded = tokenizer(tokens)
        assert list(decoded.key_signatures) == [KeySignature(0, 3, 1)]

    # Compound tokenizations do not support key signatures and must disable it
    for tokenization in ("CPWord", "Octuple", "MuMIDI", "Structured"):
        config = miditok.TokenizerConfig(use_key_signatures=True)
        tokenizer = getattr(miditok, tokenization)(tokenizer_config=config)
        assert not tokenizer.config.use_key_signatures


@pytest.mark.parametrize("tokenization", ["REMI", "TSD", "MIDILike", "PerTok"])
@pytest.mark.parametrize("control_change_n_bins", [3, 128])
@pytest.mark.parametrize("use_programs", [False, True])
def test_control_changes_round_trip(
    tokenization: str, control_change_n_bins: int, use_programs: bool, tmp_path: Path
) -> None:
    """Check CC value categories, command ordering and saved-tokenizer round trips."""
    score = Score(480)
    track = Track(program=0, is_drum=False, name="control_changes")
    track.notes.append(Note(0, 240, 60, 100))
    track.notes.append(Note(480, 240, 62, 100))
    # Parameter selection must precede data entry; both increments must survive.
    track.controls.extend(
        [
            ControlChange(0, 101, 0),
            ControlChange(0, 100, 5),
            ControlChange(0, 6, 31),
            ControlChange(0, 38, 42),
            ControlChange(0, 96, 0),
            ControlChange(0, 96, 0),
            ControlChange(0, 64, 127),
            ControlChange(120, 1, 31),
            ControlChange(120, 3, 31),
            ControlChange(120, 66, 63),
            ControlChange(120, 122, 64),
            ControlChange(240, 7, 96),
            ControlChange(240, 64, 48),
            ControlChange(240, 66, 64),
            ControlChange(240, 84, 31),
            ControlChange(240, 88, 31),
            ControlChange(480, 64, 0),
            ControlChange(480, 67, 48),
        ]
    )
    expected_control_numbers = [control.number for control in track.controls]
    expected_control_values = [
        0,
        5,
        31,
        42,
        0,
        0,
        127,
        31,
        31,
        0,
        127,
        96,
        48,
        127,
        31,
        31,
        0,
        48,
    ]
    if control_change_n_bins == 3:
        expected_control_values[7] = 0
        expected_control_values[11] = 127
        expected_control_values[12] = 64
        expected_control_values[17] = 64
    score.tracks.append(track)
    # Enabling programs merges these tracks, which must not reorder their CCs.
    score.tracks.append(Track(program=0, notes=[Note(0, 240, 67, 80)]))
    score.tempos.append(Tempo(0, 120))
    score.time_signatures.append(TimeSignature(0, 4, 4))

    params = deepcopy(TOKENIZER_CONFIG_KWARGS)
    params.update(
        {
            "use_control_changes": True,
            "control_change_numbers": sorted(set(expected_control_numbers)),
            "control_change_n_bins": control_change_n_bins,
            "use_programs": use_programs,
            "use_rests": True,
            "use_tempos": True,
            "use_time_signatures": True,
            "use_sustain_pedals": False,
            "use_pitch_bends": True,
        }
    )
    adjust_tok_params_for_tests(tokenization, params)
    tokenizer = getattr(miditok, tokenization)(miditok.TokenizerConfig(**params))

    # Four continuous controllers, two switches and eight discrete controllers.
    assert len(tokenizer.token_ids_of_type("ControlChange")) == (
        4 * control_change_n_bins + 2 * 2 + 8 * 128
    )
    tokenizer.save(tmp_path)
    tokenizer_reloaded = getattr(miditok, tokenization)(
        params=tmp_path / DEFAULT_TOKENIZER_FILE_NAME
    )
    assert tokenizer_reloaded == tokenizer
    assert tokenizer_reloaded.config.control_change_n_bins == control_change_n_bins

    score_decoded, _, has_errors = tokenize_and_check_equals(
        score, tokenizer_reloaded, "control_changes"
    )
    assert not has_errors
    assert tokenizer.config.use_control_changes
    assert [cc.number for cc in score_decoded.tracks[0].controls] == (
        expected_control_numbers
    )
    assert [cc.value for cc in score_decoded.tracks[0].controls] == (
        expected_control_values
    )


@pytest.mark.parametrize("tokenization", ["REMI", "TSD", "MIDILike"])
@pytest.mark.parametrize("use_control_changes", [False, True])
@pytest.mark.parametrize("include_cc64", [False, True])
@pytest.mark.parametrize("sustain_pedal_duration", [False, True])
def test_control_changes_sustain_precedence(
    tokenization: str,
    use_control_changes: bool,
    include_cc64: bool,
    sustain_pedal_duration: bool,
) -> None:
    """Keep exactly one sustain representation, including after MIDI export."""
    score = Score(480)
    score.tracks.append(
        Track(
            notes=[Note(0, 960, 60, 80)],
            controls=[
                ControlChange(0, 101, 0),
                ControlChange(0, 100, 5),
                ControlChange(0, 6, 31),
                ControlChange(0, 96, 0),
                ControlChange(0, 96, 0),
                ControlChange(0, 1, 32),
                ControlChange(0, 64, 127),
                ControlChange(480, 64, 0),
            ],
        )
    )
    # MIDI loading provides both raw CC64 messages and derived pedal intervals.
    score = Score.from_midi(score.dumps_midi())
    config = miditok.TokenizerConfig(
        use_control_changes=use_control_changes,
        control_change_numbers=[101, 100, 6, 96, 1] + ([64] if include_cc64 else []),
        use_sustain_pedals=True,
        sustain_pedal_duration=sustain_pedal_duration,
    )
    with warnings.catch_warnings(record=True) as config_warnings:
        warnings.simplefilter("always")
        tokenizer = getattr(miditok, tokenization)(config)

    use_cc64 = use_control_changes and include_cc64
    assert len(config_warnings) == int(use_cc64)
    if use_cc64:
        assert "CC64" in str(config_warnings[0].message)
    assert tokenizer.config.use_sustain_pedals == (not use_cc64)
    assert tokenizer.config.sustain_pedal_duration == (
        sustain_pedal_duration and not use_cc64
    )

    sequences = tokenizer(score)
    assert any(
        token.startswith("Pedal") for sequence in sequences for token in sequence.tokens
    ) == (not use_cc64)
    decoded = tokenizer(sequences)
    if use_control_changes:
        assert [cc.number for cc in decoded.tracks[0].controls if cc.number != 64] == [
            101,
            100,
            6,
            96,
            96,
            1,
        ]
    assert [cc.value for cc in decoded.tracks[0].controls if cc.number == 64] == [
        127,
        0,
    ]
    reloaded = Score.from_midi(decoded.dumps_midi())
    assert [(pedal.time, pedal.duration) for pedal in reloaded.tracks[0].pedals] == [
        (0, reloaded.ticks_per_quarter)
    ]


@pytest.mark.parametrize("control_change_n_bins", [0, 1, 129])
def test_control_changes_invalid_bins(control_change_n_bins: int) -> None:
    """Reject bin counts that cannot represent the continuous CC value range."""
    with pytest.raises(ValueError, match="control_change_n_bins"):
        miditok.TokenizerConfig(
            use_control_changes=True, control_change_n_bins=control_change_n_bins
        )


@pytest.mark.parametrize("use_programs", [False, True])
@pytest.mark.parametrize("one_token_stream", [False, True])
@pytest.mark.parametrize("program_changes", [False, True])
def test_beat_tokenizer_patterns(
    use_programs: bool, one_token_stream: bool, program_changes: bool
) -> None:
    """Check explicit paper patterns, relative pitches, repeated onsets and rests."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            num_velocities=127,
            use_programs=use_programs,
            one_token_stream_for_programs=one_token_stream,
            program_changes=program_changes,
        )
    )
    assert tokenizer.config.program_changes == (program_changes and use_programs)
    assert tokenizer.config.use_programs
    assert tokenizer.config.one_token_stream_for_programs
    assert tokenizer.one_token_stream
    score = Score(480)
    score.tracks.append(
        Track(
            notes=[
                Note(0, 960, 60, 80),
                Note(120, 120, 67, 100),
                Note(360, 120, 67, 60),
                Note(1920, 480, 64, 80),
            ]
        )
    )
    expected = [
        "Bar_None",
        "TimeSig_4/4",
        "Beat_None",
        "Program_0",
        "Pitch_67",
        "Pattern_10",
        "Velocity_80",
        "Pitch_7",
        "Pattern_53",
        "Velocity_80",
        "Beat_None",
        "Program_0",
        "Pitch_60",
        "Pattern_80",
        "Velocity_80",
        "Beat_None",
        "Rest_None",
        "Beat_None",
        "Rest_None",
        "Bar_None",
        "Beat_None",
        "Program_0",
        "Pitch_64",
        "Pattern_53",
        "Velocity_80",
    ]
    tokens = tokenizer(score)
    assert isinstance(tokens, miditok.TokSequence)
    assert tokens.tokens == expected
    assert tokenizer.tokens_errors(tokens) == 0
    decoded = tokenizer(tokens).resample(480)
    expected_notes = [
        Note(0, 960, 60, 80),
        Note(120, 120, 67, 80),
        Note(360, 120, 67, 80),
        Note(1920, 480, 64, 80),
    ]
    assert list(decoded.tracks[0].notes) == expected_notes
    assert "Step" not in tokenizer.tokens_types_graph
    assert "Duration" not in tokenizer.tokens_types_graph
    assert len(tokenizer.token_ids_of_type("Pattern")) == 81
    assert tokenizer.preprocess_score(score) == tokenizer.preprocess_score(
        tokenizer.preprocess_score(score)
    )


def test_beat_multitrack_and_meters(tmp_path: Path) -> None:
    """Preserve sustain across meter changes and order instruments per beat."""
    config = miditok.TokenizerConfig(
        use_programs=True,
        use_tempos=True,
        use_time_signatures=True,
        time_signature_range={8: [3, 6], 4: [4], 2: [2]},
        num_velocities=127,
    )
    tokenizer = miditok.BEAT(config)
    score = Score(480)
    score.time_signatures = [
        TimeSignature(0, 3, 8),
        TimeSignature(1440, 6, 8),
        TimeSignature(2880, 2, 2),
    ]
    score.tempos = [Tempo(0, 120), Tempo(240, 90), Tempo(300, 100)]
    score.tracks = [
        Track(program=24, notes=[Note(0, 4800, 60, 80)]),
        Track(program=0, notes=[Note(0, 240, 72, 90), Note(1440, 480, 72, 90)]),
        Track(is_drum=True, notes=[Note(0, 120, 36, 100)]),
    ]
    original = score.copy()
    tokens = tokenizer(score)
    assert score == original
    first_beat = tokens.tokens[tokens.tokens.index("Beat_None") + 1 :]
    first_beat = first_beat[: first_beat.index("Beat_None")]
    assert [token for token in first_beat if token.startswith("Program_")] == [
        "Program_0",
        "Program_24",
        "Program_-1",
    ]
    decoded, expected, has_errors = tokenize_and_check_equals(
        score, tokenizer, "beat_meters"
    )
    assert not has_errors
    guitar = next(track for track in decoded.tracks if track.program == 24)
    assert guitar.notes[0].duration == 10 * decoded.ticks_per_quarter
    assert [tempo.time for tempo in expected.tempos] == [
        0,
        6 * expected.ticks_per_quarter // 4,
    ]
    tokenizer.save(tmp_path)
    restored = miditok.BEAT(params=tmp_path / DEFAULT_TOKENIZER_FILE_NAME)
    assert restored(score).tokens == tokens.tokens


@pytest.mark.parametrize("use_velocities", [False, True])
def test_beat_overlaps_and_long_sustain(use_velocities: bool) -> None:
    """Truncate overlapping pitches and sustain beyond duration-vocabulary limits."""
    tokenizer = miditok.BEAT(miditok.TokenizerConfig(use_velocities=use_velocities))
    score = Score(4)
    score.tracks.append(
        Track(
            notes=[
                Note(0, 100, 60, 80),
                Note(0, 2, 60, 80),
                Note(4, 8, 60, 80),
                Note(0, 100, 72, 80),
            ]
        )
    )
    preprocessed = tokenizer.preprocess_score(score)
    assert [
        (note.time, note.duration, note.pitch) for note in preprocessed.tracks[0].notes
    ] == [(0, 4, 60), (0, 100, 72), (4, 8, 60)]
    _, _, has_errors = tokenize_and_check_equals(score, tokenizer, "beat_overlaps")
    assert not has_errors


def test_beat_empty_score() -> None:
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(use_tempos=True, use_time_signatures=True)
    )
    score = Score(480)
    tokens = tokenizer(score)
    assert tokens.tokens[-2:] == ["Beat_None", "Rest_None"]
    assert not tokenizer(tokens).tracks


def test_beat_grid_splitting() -> None:
    """Split sustain-only passages into six eighth-note beats per 6/8 bar."""
    tokenizer = miditok.BEAT(miditok.TokenizerConfig(use_time_signatures=True))
    score = Score(4)
    score.time_signatures = [TimeSignature(0, 6, 8)]
    score.tracks = [Track(notes=[Note(0, 36, 60, 80)])]
    sequence = tokenizer(score)
    beats = sequence.split_per_beats()
    bars = sequence.split_per_bars()
    assert len(beats) == 18
    assert len(bars) == 3
    assert all(beat.tokens.count("Beat_None") == 1 for beat in beats)
    assert all(bar.tokens.count("Beat_None") == 6 for bar in bars)
    assert [token for beat in beats for token in beat.tokens] == sequence.tokens


def test_beat_default_single_stream() -> None:
    """Encode program tokens in a single stream with the default configuration."""
    tokenizer = miditok.BEAT()
    score = Score(4)
    score.tracks = [
        Track(program=40, notes=[Note(0, 4, 72, 90)]),
        Track(program=0, notes=[Note(0, 4, 60, 90)]),
    ]
    tokens = tokenizer(score)
    assert isinstance(tokens, miditok.TokSequence)
    decoded = tokenizer.decode(tokens)
    assert [(track.program, track.notes[0].pitch) for track in decoded.tracks] == [
        (0, 60),
        (40, 72),
    ]


@pytest.mark.parametrize("is_drum", [False, True])
@pytest.mark.parametrize("use_pitchdrum_tokens", [False, True])
@pytest.mark.parametrize("use_velocities", [False, True])
@pytest.mark.parametrize("time_signature", [(4, 4), (6, 8)])
def test_beat_same_program_tracks(
    is_drum: bool,
    use_pitchdrum_tokens: bool,
    use_velocities: bool,
    time_signature: tuple[int, int],
    tmp_path: Path,
) -> None:
    """Preserve track identity through overlapping pitches, rests and later onsets."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            num_velocities=127,
            use_velocities=use_velocities,
            use_pitchdrum_tokens=use_pitchdrum_tokens,
            use_time_signatures=True,
        )
    )
    score = Score(time_signature[1])
    score.time_signatures = [TimeSignature(0, *time_signature)]
    score.tracks = [
        # The first track starts after the second, then falls silent and returns.
        Track(is_drum=is_drum, notes=[Note(4, 4, 60, 80), Note(24, 4, 64, 80)]),
        Track(program=24, notes=[Note(0, 4, 72, 90)]),
        Track(is_drum=is_drum, notes=[Note(0, 20, 60, 100)]),
        Track(is_drum=is_drum, notes=[Note(12, 4, 60, 70), Note(24, 4, 60, 70)]),
    ]
    original = score.copy()
    preprocessed = tokenizer.preprocess_score(score)
    assert score == original
    assert len(preprocessed.tracks) == 4
    assert tokenizer.preprocess_score(preprocessed) == preprocessed

    tokens = tokenizer(score)
    assert isinstance(tokens, miditok.TokSequence)
    assert tokenizer.tokens_errors(tokens) == 0
    program_token = "Program_-1" if is_drum else "Program_0"
    beats = tokens.split_per_beats()
    assert len(beats) == 7
    assert all(beat.tokens.count(program_token) == 3 for beat in beats)
    # Beat six is silent for all three tracks; their slots must still be present.
    assert beats[5].tokens == ["Beat_None", *([program_token, "Rest_None"] * 3)]
    decoded = tokenizer(tokens).resample(score.ticks_per_quarter)
    expected_tracks = sorted(
        preprocessed.tracks, key=lambda track: 128 if track.is_drum else track.program
    )
    assert len(decoded.tracks) == len(expected_tracks)
    for expected, actual in zip(expected_tracks, decoded.tracks, strict=True):
        assert (actual.program, actual.is_drum) == (expected.program, expected.is_drum)
        if not use_velocities:
            for note in expected.notes:
                note.velocity = miditok.constants.DEFAULT_VELOCITY
        assert actual.notes == expected.notes

    tokenizer.save(tmp_path)
    restored = miditok.BEAT(params=tmp_path / DEFAULT_TOKENIZER_FILE_NAME)
    assert restored.one_token_stream
    assert restored(score).tokens == tokens.tokens
    assert restored(restored(score)).tracks == tokenizer(tokens).tracks


@pytest.mark.parametrize("use_tempos", [False, True])
def test_beat_key_signatures(use_tempos: bool, tmp_path: Path) -> None:
    """Align keys to bars, keep the last collision, and decode globals only once."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            use_programs=True,
            use_key_signatures=True,
            use_time_signatures=True,
            use_tempos=use_tempos,
        )
    )
    score = Score(480)
    score.time_signatures = [TimeSignature(0, 4, 4), TimeSignature(1920, 3, 4)]
    score.tempos = [Tempo(0, 120), Tempo(240, 90)]
    score.key_signatures = [
        KeySignature(0, -1, 0),
        KeySignature(240, 1, 0),
        KeySignature(480, 2, 1),
        KeySignature(1920, -3, 0),
        KeySignature(2400, 4, 1),
        KeySignature(4560, 5, 0),
    ]
    score.tracks = [
        Track(program=0, notes=[Note(0, 2400, 60, 80)]),
        Track(program=24, notes=[Note(480, 1440, 67, 90)]),
    ]
    original = score.copy()
    expected_keys = [
        KeySignature(0, -1, 0),
        KeySignature(1920, -3, 0),
        KeySignature(3360, 4, 1),
        KeySignature(4800, 5, 0),
    ]
    preprocessed = tokenizer.preprocess_score(score)
    assert score == original
    assert list(preprocessed.resample(480).key_signatures) == expected_keys
    assert tokenizer.preprocess_score(preprocessed) == preprocessed
    tokens = tokenizer(score)
    assert [token for token in tokens.tokens if token.startswith("KeySig_")] == [
        "KeySig_-1:0",
        "KeySig_-3:0",
        "KeySig_4:1",
        "KeySig_5:0",
    ]
    assert tokenizer._tokens_errors(tokens.tokens) == 0
    assert list(tokenizer(tokens).resample(480).key_signatures) == expected_keys
    tokenizer.save(tmp_path)
    restored = miditok.BEAT(params=tmp_path / DEFAULT_TOKENIZER_FILE_NAME)
    assert restored.config.use_key_signatures
    assert list(restored(restored(score)).resample(480).key_signatures) == expected_keys


@pytest.mark.parametrize(
    "time_signature", [(2, 2), (3, 4), (4, 4), (3, 8), (6, 8), (3, 16)]
)
@pytest.mark.parametrize("use_time_signatures", [False, True])
def test_beat_meter_resolution(
    time_signature: tuple[int, int], use_time_signatures: bool
) -> None:
    """Keep four steps per denominator unit and numerator beats in a complete bar."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            use_time_signatures=use_time_signatures,
            time_signature_range={2: [2], 4: [3, 4], 8: [3, 6], 16: [3]},
            num_velocities=127,
        )
    )
    score = Score(480)
    score.time_signatures = [TimeSignature(0, *time_signature)]
    assert tokenizer.config.use_time_signatures
    numerator, denominator = time_signature
    ticks_per_beat = 480 * 4 // denominator
    score.tracks = [
        Track(
            notes=[
                Note(0, ticks_per_beat * numerator, 60, 80),
                Note(0, ticks_per_beat * 3 // 4, 67, 80),
            ]
        )
    ]
    preprocessed = tokenizer.preprocess_score(score)
    assert preprocessed.ticks_per_quarter == denominator
    tokens = tokenizer(score)
    assert tokens.tokens.count("Beat_None") == numerator
    assert "Pattern_51" in tokens.tokens
    assert tokens.tokens.count("Pattern_80") == numerator - 1
    assert list(tokenizer(tokens).resample(480).tracks[0].notes) == sorted(
        score.tracks[0].notes, key=lambda note: (note.time, note.duration, note.pitch)
    )


@pytest.mark.parametrize(
    ("time_signatures", "notes"),
    [
        (
            [(0, 4, 4), (1920, 6, 8), (3360, 2, 2)],
            [(1800, 180, 60, 80), (3300, 300, 64, 80)],
        ),
        # The new half-note grid starts at 360, not at a multiple of 240 ticks.
        (
            [(0, 3, 16), (360, 2, 2), (2280, 3, 8)],
            [(330, 270, 60, 80), (360, 240, 64, 80), (2040, 300, 67, 80)],
        ),
    ],
)
def test_beat_sustain_across_denominator_changes(
    time_signatures: list[tuple[int, int, int]], notes: list[tuple[int, int, int, int]]
) -> None:
    """Use the onset and offset grids when a note spans a denominator change."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            use_time_signatures=True,
            time_signature_range={2: [2], 4: [4], 8: [3, 6], 16: [3]},
            num_velocities=127,
        )
    )
    score = Score(480)
    score.time_signatures = [TimeSignature(*time_sig) for time_sig in time_signatures]
    score.tracks = [Track(notes=[Note(*note) for note in notes])]
    preprocessed = tokenizer.preprocess_score(score)
    assert list(preprocessed.resample(480).tracks[0].notes) == list(
        score.tracks[0].notes
    )
    assert tokenizer.preprocess_score(preprocessed) == preprocessed
    tokens = tokenizer(score)
    assert tokenizer.tokens_errors(tokens) == 0
    assert list(tokenizer(tokens).resample(480).tracks[0].notes) == list(
        score.tracks[0].notes
    )


def test_beat_key_signature_before_final_meter_change() -> None:
    """Use the preceding meter to find the last bar boundary of the input score."""
    tokenizer = miditok.BEAT(
        miditok.TokenizerConfig(
            use_key_signatures=True,
            use_time_signatures=True,
        )
    )
    score = Score(480)
    score.time_signatures = [TimeSignature(0, 4, 4), TimeSignature(1920, 3, 4)]
    score.key_signatures = [KeySignature(1440, 2, 1)]
    score.tracks = [Track(notes=[Note(0, 480, 60, 80)])]
    decoded = tokenizer(tokenizer(score)).resample(480)
    assert list(decoded.key_signatures) == [
        KeySignature(0, 0, 0),
        KeySignature(1920, 2, 1),
    ]
