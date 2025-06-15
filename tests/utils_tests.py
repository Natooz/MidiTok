"""Test validation methods."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from symusic import (
    Note,
    Pedal,
    Score,
    Tempo,
    TextMeta,
    TimeSignature,
    Track,
)

import miditok
from miditok.attribute_controls import BarAttributeControl
from miditok.constants import (
    CHORD_MAPS,
    TIME_SIGNATURE,
    TIME_SIGNATURE_RANGE,
    USE_NOTE_DURATION_PROGRAMS,
)
from miditok.utils import get_bars_ticks, get_beats_ticks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from symusic.core import NoteTickList, TempoTickList

    from miditok import MusicTokenizer, TokSequence

SEED = 777

HERE = Path(__file__).parent
MIDI_PATHS_ONE_TRACK = sorted((HERE / "MIDIs_one_track").rglob("*.mid"))
MIDI_PATHS_MULTITRACK = sorted((HERE / "MIDIs_multitrack").rglob("*.mid"))
MIDI_PATHS_CORRUPTED = sorted((HERE / "MIDIs_corrupted").rglob("*.mid"))
MIDI_PATHS_ALL = sorted(
    deepcopy(MIDI_PATHS_ONE_TRACK) + deepcopy(MIDI_PATHS_MULTITRACK)
)
ABC_PATHS = sorted((HERE / "abc_files").rglob("*.abc"))
TEST_LOG_DIR = HERE / "test_logs"
# MIDI files known to contain tricky contents (time sig, pedals...) and edge case
# situations, likely to make some tests fail.
MIDIS_ONE_TRACK_HARD_NAMES = [
    "6338816_Etude No. 4.mid",
    "6354774_Macabre Waltz.mid",
    "Maestro_9.mid",
    "POP909_191.mid",
]
MIDI_PATHS_ONE_TRACK_HARD = [
    path for path in MIDI_PATHS_ONE_TRACK if path.name in MIDIS_ONE_TRACK_HARD_NAMES
]

# TOKENIZATIONS
ALL_TOKENIZATIONS = miditok.tokenizations.__all__
TOKENIZATIONS_TRAIN = ["REMI", "TSD", "MMM"]
TRAINING_MODELS = ["BPE", "Unigram", "WordPiece"]

# TOK CONFIG PARAMS
TIME_SIGNATURE_RANGE_TESTS = TIME_SIGNATURE_RANGE
TIME_SIGNATURE_RANGE_TESTS.update({2: [2, 3, 4]})
TIME_SIGNATURE_RANGE_TESTS[4].append(8)
TOKENIZER_CONFIG_KWARGS = {
    "special_tokens": ["PAD", "BOS_None", "EOS", "EOS-test_None"],
    "beat_res": {(0, 4): 8, (4, 12): 4, (12, 16): 2},
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "num_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": TIME_SIGNATURE_RANGE_TESTS,
    "chord_maps": CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
    "use_bar_end_tokens": True,
}
MAX_BAR_EMBEDDING = 2000

# For attribute controls
_TOKEN_TYPES_TRACK_ATTRIBUTE = {"Track", "Program", "ProgramChange"}
_TOKEN_TYPES_BREAK_ITERATION = {
    "Bar",
    "TimeShift",
    "TimeSig",
    "Tempo",
    "Pitch",
    "NoteOn",
}
TRACKS_RANDOM_RATIO_RANGE = (0.2, 0.7)
BARS_RANDOM_RATIO_RANGE = (0.2, 0.7)


def adjust_tok_params_for_tests(tokenization: str, params: dict[str, Any]) -> None:
    """
    Adjust tokenizer config parameters for tests.

    Depending on the tokenization, some adjustments are necessary to ensure that the
    Score decoded from tokens is identical to the original one.

    :param tokenization: tokenization.
    :param params: parameters as a dictionary of keyword arguments.
    """
    # Increase the TimeShift voc for Structured as it doesn't support successive
    # TimeShifts.
    if tokenization == "Structured":
        params["beat_res"] = {(0, 512): 8}
    # We don't test time signatures with Octuple as it can lead to time shifts, as the
    # TS changes are only carried at the onset times of the notes.
    elif tokenization in ["Octuple", "MuMIDI"]:
        params["max_bar_embedding"] = MAX_BAR_EMBEDDING
        if tokenization == "Octuple":
            params["use_time_signatures"] = False
    # Rests and time sig can mess up with CPWord, when a Rest that is crossing new bar
    # is followed by a new TimeSig change, as TimeSig are carried with Bar tokens (and
    # there is None is this case).
    elif (
        tokenization == "CPWord"
        and params.get("use_time_signatures", False)
        and params.get("use_rests", False)
    ):
        params["use_rests"] = False
    # PerTok needs a number of separate args
    elif tokenization == "PerTok":
        params["beat_res"] = {(0, 128): 4, (0, 32): 3}
        params["use_microtiming"] = True
        params["ticks_per_quarter"] = 220

        # params["beat_res"] = {(0, 16): 4}
        # params["use_microtiming"] = False
        # params["ticks_per_quarter"] = 16

        params["max_microtiming_shift"] = 0.25
        params["num_microtiming_bins"] = 110
        params["use_rests"] = False
        params["use_sustain_pedals"] = False
        params["use_bar_end_tokens"] = False
        params["use_tempos"] = False
        params["use_time_signatures"] = True
        params["use_pitch_bends"] = False
        params["use_pitchdrum_tokens"] = False
        params["use_pitch_intervals"] = False


def sort_score(score: Score, sort_tracks: bool = True) -> None:
    """
    Sorts a Score: its notes and other track events, and the tracks themselves.

    :param score: ``symusic.Score`` object to sort.
    :param sort_tracks: will sort the tracks by program if given True.
    """
    for track in score.tracks:
        if track.is_drum:
            track.program = 0  # need to be done before sorting tracks per program
        # notes sorted by (start, duration, pitch) by symusic
        # we keep this order here as if we need to sort them specifically, this will be
        # done in tokenizer.preprocess_score
        track.notes.sort()
        # track.notes.sort(key=lambda x: (x.time, x.pitch, x.duration, x.velocity))
        # track.pedals.sort()
        # track.pitch_bends.sort()
        # track.controls.sort()

    # Sorts tracks
    # Score decoded with one_token_stream contains tracks sorted by note occurrence
    # This is done at the end as we may
    if sort_tracks:
        score.tracks.sort(key=lambda x: (x.program, x.is_drum))


def adapt_ref_score_before_tokenize(score: Score, tokenizer: MusicTokenizer) -> None:
    """
    Adapt (inplace) the contents of a Score before it is tokenized.

    :param score: ``symusic.Score`` object to adapt.
    :param tokenizer: tokenizer being used.
    """
    tokenization = type(tokenizer).__name__ if tokenizer is not None else None

    if tokenizer._note_on_off:
        # Need to sort the notes with all these keys, as otherwise some velocity values
        # might be mixed up for notes with the same onset and duration values as the
        # tokens are decoded in a FIFO logic.
        # But before sorting, we need to merge the tracks if needed, and clip durations
        if (
            tokenizer.config.use_programs
            and tokenizer.config.one_token_stream_for_programs
        ):
            miditok.utils.merge_same_program_tracks(score.tracks)

        # If a max_duration is provided, we clip the durations of the notes before
        # tokenizing, otherwise these notes will be tokenized with durations > to this
        # limit, which would yield errors when checking TSE.
        if "max_duration" in tokenizer.config.additional_params:
            max_durations = np.array(
                [
                    [
                        end_tick,
                        tokenizer._time_token_to_ticks(
                            tokenizer.config.additional_params["max_duration"],
                            tpb,
                        ),
                    ]
                    for end_tick, tpb in miditok.utils.get_score_ticks_per_beat(score)
                ],
                dtype=np.intc,
            )
            for track in score.tracks:
                clip_durations(track.notes, max_durations)

        # This is required for tests, as after resampling, for some overlapping notes
        # with different onset times with the second note ending last (as FIFO), it can
        # happen that after resampling the second note now ends before the first one.
        # Example: POP909_191 at tick 3152 (time division of 16 tpq)
        for track in score.tracks:
            miditok.utils.fix_offsets_overlapping_notes(track.notes)

        # Now we can sort the notes
        sort_score(score, sort_tracks=False)

    # For Octuple, CPWord and MMM, the time signature is carried with the notes.
    # If a Score doesn't have any note, no time signature will be tokenized, and in turn
    # decoded. If that's the case, we simply set time signatures to the default one.
    if (
        tokenizer.config.use_time_signatures
        and tokenization in ["Octuple", "CPWord", "MMM"]
        and (len(score.tracks) == 0 or len(score.tracks[0].notes) == 0)
    ):
        score.time_signatures = [TimeSignature(0, *TIME_SIGNATURE)]


def adapt_ref_score_for_tests_assertion(
    score: Score, tokenizer: MusicTokenizer
) -> Score:
    """
    Adapt a reference raw/unprocessed Score for test assertions.

    This method is meant to be used with a reference Score, and preprocess it so that
    its contents match exactly those of the Score decoded from the tokens of this
    reference Score.
    The transformed Score will be preprocessed (`tokenizer.preprocess_score()`), and
    other attributes such as tempos or time signature times may be altered.

    :param score: ``symusic.Score`` object reference.
    :param tokenizer: in order to downsample the Score before sorting its content.
    :return: a new ``symusic.Score`` object with track (and notes) sorted.
    """
    tokenization = type(tokenizer).__name__ if tokenizer is not None else None

    # Preprocess the Score: downsample it, remove notes outside of pitch range...
    score = tokenizer.preprocess_score(score)

    # For Octuple, as tempo is only carried at notes times, we need to adapt
    # their times for comparison. Set tempo changes at onset times of notes.
    # We use the first track only, as it is the one for which tempos are decoded
    if tokenizer.config.use_tempos and tokenization in ["Octuple"]:
        if len(score.tracks) > 0:
            adapt_tempo_changes_times(
                score.tracks
                if tokenizer.config.one_token_stream_for_programs
                else score.tracks[:1],
                score.tempos,
                tokenizer.default_tempo,
            )
        else:
            score.tempos = [Tempo(0, tokenizer.default_tempo)]

    # If tokenizing each track independently without using Program tokens, the tokenizer
    # will have no way to know the original program of each token sequence when decoding
    # them. We thus resort to set the program of the original score to the default value
    # (0, piano) to match the decoded Score. The content of the tracks (notes, controls)
    # will still be checked.
    if (
        not tokenizer.config.one_token_stream_for_programs
        and not tokenizer.config.use_programs
    ):
        for track in score.tracks:
            track.program = 0
            track.is_drum = False

    return score


def scores_notes_equals(
    score1: Score,
    score2: Score,
    check_velocities: bool,
    use_note_duration_programs: Sequence[int],
    use_time_range: bool = False,
) -> list[tuple[int, str, list[tuple[str, Note | int, int]]]]:
    """
    Check that the notes from two Scores are all equal.

    If they are not all equal, the method returns the list of errors.

    :param score1: first ``symusic.Score``.
    :param score2: second ``symusic.Score``.
    :param check_velocities: whether to check velocities of notes.
    :param use_note_duration_programs: programs for which the note durations are
        tokenized. This is used to determine whether to assert note durations equality.
    :return: list of errors.
    """
    if len(score1.tracks) != len(score2.tracks):
        return [(0, "num tracks", [])]
    errors = []
    for track1, track2 in zip(score1.tracks, score2.tracks):
        if track1.program != track2.program or track1.is_drum != track2.is_drum:
            errors.append((0, "program", [track1.program, track2.program]))
            continue
        if len(track1.notes) != len(track2.notes):
            errors.append((0, "num notes", []))
            continue
        track_program = -1 if track1.is_drum else track1.program
        using_note_durations = track_program in use_note_duration_programs
        # Need to set the note durations of the first track to the durations of the
        # second and sort the notes. Without duration tokens, the order of the notes
        # may be altered as during preprocessing they are order by pitch and duration.
        if not using_note_durations:
            notes2 = track2.notes.numpy()
            notes2["duration"] = track1.notes.numpy()["duration"]
            track1.notes = Note.from_numpy(**notes2)
            track1.notes.sort(key=lambda n: (n.time, n.pitch, n.duration, n.velocity))
        track_errors = tracks_notes_equals(
            track1,
            track2,
            check_velocities,
            using_note_durations,
            use_time_range,
            max_time_range=int(score1.ticks_per_quarter * 0.5),
        )
        if len(track_errors) > 0:
            errors.append((track1.program, track1.name, track_errors))
    return errors


def tracks_notes_equals(
    track1: Track,
    track2: Track,
    check_velocities: bool = True,
    check_durations: bool = True,
    use_time_range: bool = False,
    max_time_range: int = 220,
) -> list[tuple[str, Note | int, int]]:
    if not use_time_range:
        errors = []
        for note1, note2 in zip(track1.notes, track2.notes):
            err = notes_equals(
                note1,
                note2,
                check_velocities,
                check_durations,
            )
            if err != "":
                errors.append((err, note2, getattr(note1, err)))
        return errors
    # Sliding window search of nearby notes for hires tokenizers
    return notes_in_sliding_window_equals(
        track1.notes,
        track2.notes,
        check_velocities=check_velocities,
        check_durations=check_durations,
        max_time_range=max_time_range,
    )


def notes_in_sliding_window_equals(
    notes_1: NoteTickList,
    notes_2: NoteTickList,
    check_velocities: bool = True,
    check_durations: bool = True,
    max_time_range: int = 120,
) -> list[tuple[str, Note | int, int]]:
    errors = []
    for idx, note_1 in enumerate(notes_1):
        potential_notes = get_notes_in_range(idx=idx, note_list=notes_2, window_size=25)
        potential_notes = [
            note for note in potential_notes if note.pitch == note_1.pitch
        ]
        if potential_notes is None:
            errors.append(("pitch", notes_2[idx], note_1.pitch))
            continue

        if not any(
            (abs(note_1.start - note_2.start) < max_time_range)
            for note_2 in potential_notes
        ):
            errors.append(("start", notes_2[idx], note_1.start))
            continue

        if check_durations and not any(
            (abs(note_1.end - note_2.end) < max_time_range)
            for note_2 in potential_notes
        ):
            errors.append(("end", notes_2[idx], note_1.end))
            continue

        if check_velocities and not any(
            (note_1.velocity == note_2.velocity) for note_2 in potential_notes
        ):
            errors.append(("velocity", notes_2[idx], note_1.velocity))
            continue

    return errors


def get_notes_in_range(
    idx: int, note_list: NoteTickList, window_size: int = 5
) -> NoteTickList:
    start = max(0, idx - window_size)
    end = min(len(note_list) - 1, idx + window_size)
    return note_list[start : end + 1]


def notes_equals(
    note1: Note,
    note2: Note,
    check_velocity: bool = True,
    check_duration: bool = True,
) -> str:
    if note1.start != note2.start:
        return "start"
    if check_duration and note1.end != note2.end:
        return "end"
    if note1.pitch != note2.pitch:
        return "pitch"
    if check_velocity and note1.velocity != note2.velocity:
        return "velocity"
    return ""


def tempos_equals(tempos1: TempoTickList, tempos2: TempoTickList) -> bool:
    if len(tempos1) != len(tempos2):
        return False
    for tempo1, tempo2 in zip(tempos1, tempos2):
        if (
            tempo1.time != tempo2.time
            or round(tempo1.qpm, 2) != round(tempo2.qpm, 2)
            or abs(tempo1.mspq - tempo2.mspq) > 1
        ):
            return False
    return True


def check_scores_equals(
    score1: Score,
    score2: Score,
    check_velocities: bool = True,
    use_note_duration_programs: Sequence[int] = USE_NOTE_DURATION_PROGRAMS,
    check_tempos: bool = True,
    check_time_signatures: bool = True,
    check_pedals: bool = True,
    check_pitch_bends: bool = True,
    log_prefix: str = "",
    use_time_ranges: bool = False,
    max_time_range: int = 120,
) -> bool:
    has_errors = False
    types_of_errors = []

    # Checks notes and add markers if errors
    errors = scores_notes_equals(
        score1,
        score2,
        check_velocities,
        use_note_duration_programs,
        use_time_ranges,
    )

    if len(errors) > 0:
        has_errors = True
        for e, track_err in enumerate(errors):
            if track_err[1] != "num_notes":
                for err, note, exp in track_err[-1]:
                    score2.markers.append(
                        TextMeta(
                            note.start,
                            f"{e}: with note {err} (pitch {note.pitch}), expected "
                            f"{exp}",
                        )
                    )
        num_errors = sum(len(t[2]) for t in errors)
        print(f"{log_prefix} failed to encode/decode NOTES ({num_errors} errors)")

    # Check pedals
    if check_pedals:
        for inst1, inst2 in zip(score1.tracks, score2.tracks):
            if not use_time_ranges and inst1.pedals != inst2.pedals:
                types_of_errors.append("PEDALS")
                break
            inst1_pedals, inst2_pedals = inst1.pedals, inst2.pedals
            for pedal_0, pedal_1 in zip(inst1_pedals, inst2_pedals):
                if (pedal_0.time - pedal_1.time) > max_time_range or (
                    pedal_0.duration - pedal_1.duration
                ) > max_time_range:
                    types_of_errors.append("PEDALS")
                    break

    # Check pitch bends
    if check_pitch_bends:
        for inst1, inst2 in zip(score1.tracks, score2.tracks):
            if inst1.pitch_bends != inst2.pitch_bends:
                types_of_errors.append("PITCH BENDS")
                break

    """# Check control changes
    if check_control_changes:
        for inst1, inst2 in zip(score1.tracks, score2.tracks):
            if inst1.controls != inst2.controls:
                types_of_errors.append("CONTROL CHANGES")
                break"""

    # Checks tempos
    if check_tempos and not tempos_equals(score1.tempos, score2.tempos):
        types_of_errors.append("TEMPOS")

    # Checks time signatures
    if check_time_signatures:
        if not use_time_ranges and score1.time_signatures != score2.time_signatures:
            types_of_errors.append("TIME SIGNATURES")
        elif use_time_ranges:
            time_sigs1, time_sigs2 = score1.time_signatures, score2.time_signatures
            for time_sig1, time_sig2 in zip(time_sigs1, time_sigs2):
                if abs(time_sig1.time - time_sig2.time) > max_time_range:
                    types_of_errors.append("TIME SIGNATURES")
                    break

    # Prints types of errors
    has_errors = has_errors or len(types_of_errors) > 0
    for err_type in types_of_errors:
        print(f"{log_prefix} failed to encode/decode {err_type}")

    return not has_errors


def tokenize_and_check_equals(
    score: Score,
    tokenizer: MusicTokenizer,
    file_name: str,
) -> tuple[Score, Score, bool]:
    tokenization = type(tokenizer).__name__
    log_prefix = f"{file_name} / {tokenization}"
    use_time_ranges = bool(tokenization in ["PerTok"])

    # Tokenize and detokenize
    adapt_ref_score_before_tokenize(score, tokenizer)
    tokens = tokenizer(score)
    score_decoded = tokenizer(tokens)

    # Post-process the reference and decoded Scores
    # We might need to resample the original preprocessed Score, as it has been
    # resampled with its highest ticks/beat whereas the tokens has been decoded with
    # the tokenizer's time division, which can be different if using time signatures.
    score = adapt_ref_score_for_tests_assertion(score, tokenizer)
    if score.ticks_per_quarter != score_decoded.ticks_per_quarter:
        score = score.resample(tpq=score_decoded.ticks_per_quarter)
    # if not use_time_ranges:
    #     sort_score(score)
    #     sort_score(score_decoded)
    sort_score(score)
    sort_score(score_decoded)

    # Check decoded Score is identical to the reference one
    scores_equals = check_scores_equals(
        score,
        score_decoded,
        check_velocities=tokenizer.config.use_velocities,
        use_note_duration_programs=tokenizer.config.use_note_duration_programs,
        check_tempos=tokenizer.config.use_tempos and tokenization != "MuMIDI",
        check_time_signatures=tokenizer.config.use_time_signatures,
        check_pedals=tokenizer.config.use_sustain_pedals,
        check_pitch_bends=tokenizer.config.use_pitch_bends,
        log_prefix=log_prefix,
        use_time_ranges=use_time_ranges,
    )

    # Checks types and values conformity following the rules
    err_tse = tokenizer.tokens_errors(tokens)
    if isinstance(err_tse, list):
        err_tse = sum(err_tse)
    if err_tse != 0.0:
        scores_equals = False
        print(f"{log_prefix} Validation of tokens types / values successions failed")

    return score_decoded, score, not scores_equals


def del_invalid_time_sig(
    time_sigs: list[TimeSignature], time_sigs_tokenizer: list[TimeSignature]
) -> None:
    r"""
    Delete time signatures of a Score outside those supported by a tokenizer.

    This is actually unused in our tokenization test pipeline, as removing the
    invalid time signature is already done in ``preprocess_score``.

    :param time_sigs: time signatures to filter
    :param time_sigs_tokenizer:
    """
    idx = 0
    while idx < len(time_sigs):
        if (
            time_sigs[idx].numerator,
            time_sigs[idx].denominator,
        ) not in time_sigs_tokenizer:
            del time_sigs[idx]
        else:
            idx += 1


def adapt_tempo_changes_times(
    tracks: list[Track],
    tempo_changes: list[Tempo],
    default_tempo: int,
) -> None:
    r"""
    Align the times of tempo changes on those of reference notes.

    This is needed to pass the tempo tests for Octuple as the tempos
    will be decoded only from the notes.

    :param tracks: tracks of the Score to adapt the tempo changes
    :param tempo_changes: tempo changes to adapt
    :param default_tempo: default tempo value to mock at beginning if needed
    """
    times = []
    for track in tracks:
        times += [note.time for note in track.notes]
    times.sort()

    # Fixes the first tempo at the time of the first note and mock if needed
    if round(tempo_changes[0].tempo, 2) == default_tempo:
        tempo_changes[0].time = 0
    # In case the first tempo is not the default one and occurs before the first time
    # we need to shift it and mock with the default tempo value (done below)
    elif tempo_changes[0].time < times[0]:
        tempo_changes[0].time = times[0]
    if tempo_changes[0].time != 0:
        tempo_changes.insert(0, Tempo(0, default_tempo))

    time_idx = tempo_idx = 0
    while tempo_idx < len(tempo_changes):
        # Delete tempos after the last note
        if tempo_changes[tempo_idx].time > times[-1]:
            del tempo_changes[tempo_idx]
            continue
        # Loop over incoming notes to adapt times
        # Except for the first one which is at 0
        if tempo_idx > 0:
            for n, time in enumerate(times[time_idx:]):
                if time >= tempo_changes[tempo_idx].time:
                    tempo_changes[tempo_idx].time = time
                    time_idx += n
                    break
        # Delete successive tempos at the same position (keep the latest)
        if (
            tempo_idx > 0
            and tempo_changes[tempo_idx].time == tempo_changes[tempo_idx - 1].time
        ):
            del tempo_changes[tempo_idx - 1]
            continue
        tempo_idx += 1


def clip_durations(
    notes_pedals: list[Note] | list[Pedal],
    max_durations: np.ndarray,
) -> None:
    """
    Clip the duration of notes or pedals to a specific limit.

    This method is applied in the tokenization tests to a preprocessed reference Score
    to make sure that the there are no note/pedal durations that exceed the limit, as
    the Score decoded from tokens will have limited durations. This applies to
    tokenizers using *NoteOff* and *PedalOff* tokens, as otherwise (i.e. using
    *Duration* tokens) the durations are already clipped during the preprocessing step.

    :param notes_pedals: list of Note or Pedal objects to adapt.
    :param max_durations: max duration values, per tick section, as a numpy array of
        shape ``(N,2)`` for ``N`` sections, and the second dimension corresponding to
        the end tick and the max duration in ticks of each section. Processing by
        sections is required as the original maximum duration is given in beats, and
        the length of the beats of a Score can very with time signature changes.
    """
    tpb_idx = 0
    for note_pedal in notes_pedals:
        if note_pedal.time > max_durations[tpb_idx, 0]:
            tpb_idx += 1
        if note_pedal.duration > max_durations[tpb_idx, 1]:
            note_pedal.duration = max_durations[tpb_idx, 1]


def check_control_tokens_are_well_inserted(
    tokenizer: MusicTokenizer,
    score: Score,
    tokens: TokSequence | Sequence[TokSequence],
    ac_indexes: Mapping[int, Mapping[int, bool | Sequence[int]]],
) -> list[tuple[int, str]]:
    errors = []

    # If MMM split the token sequence per track
    if isinstance(tokenizer, miditok.MMM):
        tokens = tokenizer.split_tokseq_per_track(tokens, keep_track_tokens=True)

    ticks_bars = get_bars_ticks(score, only_notes_onsets=True)
    ticks_beats = get_beats_ticks(score, only_notes_onsets=True)
    for track_idx, acs in ac_indexes.items():
        for ac_idx, tracks_bars_idx in acs.items():
            controls = tokenizer.attribute_controls[ac_idx].compute(
                score.tracks[track_idx],
                score.ticks_per_quarter,
                ticks_bars,
                ticks_beats,
                tracks_bars_idx,
            )
            seq = tokens[track_idx]
            if isinstance(tokenizer.attribute_controls[ac_idx], BarAttributeControl):
                tokens_times = np.array([event.time for event in seq.events])
                for control in controls:
                    bar_tick = control.time
                    control_token = str(control)
                    token_bar_idx = np.where(tokens_times >= bar_tick)[0][0].item()
                    offset = 0
                    while (
                        token_bar_idx + offset < len(seq.tokens)
                        and seq.events[token_bar_idx + offset].time == bar_tick
                        and seq.tokens[token_bar_idx + offset] != control_token
                    ):
                        offset += 1
                    if seq.tokens[token_bar_idx + offset] != control_token:
                        errors.append(
                            (
                                track_idx,
                                f"bar-{bar_tick}_{control_token}",
                            )
                        )

            # Track-level attribute control
            # Just make sure the controls are present at the beginning (exc. MMM)
            else:
                controls_tokens = [str(control) for control in controls]
                for control in controls_tokens:
                    for token in seq.tokens:
                        if token == control:
                            break
                        if token in controls_tokens:
                            continue
                        token_type = token.split("_")[0]
                        if not token_type.startswith("ACTrack") and (
                            token_type not in _TOKEN_TYPES_TRACK_ATTRIBUTE
                            or token_type in _TOKEN_TYPES_BREAK_ITERATION
                        ):
                            errors.append((track_idx, control))
                            break

    return errors
