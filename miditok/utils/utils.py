"""Useful methods."""
from __future__ import annotations

import warnings
from copy import copy
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from symusic import (
    ControlChange,
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TextMeta,
    TimeSignature,
    Track,
)
from symusic.core import NoteTickList

from miditok.classes import Event
from miditok.constants import (
    DRUM_PITCH_RANGE,
    INSTRUMENT_CLASSES,
    MIDI_INSTRUMENTS,
    PITCH_CLASSES,
    TIME_SIGNATURE,
    UNKNOWN_CHORD_PREFIX,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miditoolkit import MidiFile
    from symusic.core import TrackTickList


def convert_ids_tensors_to_list(ids) -> list[int] | list[list[int]]:  # noqa: ANN001
    """
    Convert a PyTorch, Tensorflow Tensor or numpy array to a list of integers.

    This method works with Jax too. It is recursive and will convert nested
    Tensors/arrays.

    :param ids: ids sequence to convert.
    :return: the input, as a list of integers
    """
    # Convert tokens to list if necessary
    if not isinstance(ids, list):
        if type(ids).__name__ in ["Tensor", "EagerTensor"]:
            ids = ids.numpy()
        if not isinstance(ids, np.ndarray):
            msg = (
                "The tokens must be given as a list of integers, np.ndarray, or"
                "PyTorch/Tensorflow tensor"
            )
            raise TypeError(msg)
        ids = ids.astype(int).tolist()
    else:
        # Recursively checks the content are ints (only check first item)
        el = ids[0]
        while isinstance(el, list):
            el = el[0] if len(el) > 0 else None

        # Check endpoint type
        if el is None:
            pass
        elif not isinstance(el, int):
            # Recursively try to convert elements of the list
            for ei in range(len(ids)):
                ids[ei] = convert_ids_tensors_to_list(ids[ei])

    return ids


def get_midi_programs(midi: Score) -> list[tuple[int, bool]]:
    r"""
    Return the list of programs of the tracks of a MIDI.

    :param midi: the MIDI object to extract tracks programs
    :return: the list of track programs, as a list of tuples (program, is_drum)
    """
    return [(int(track.program), track.is_drum) for track in midi.tracks]


def remove_duplicated_notes(
    notes: NoteTickList | dict[str, np.ndarray], consider_duration: bool = False
) -> None:
    r"""
    Remove (inplace) duplicated notes, i.e. with the same pitch and onset time.

    This can be done either from a ``symusic.NoteTickList``, or a note structure of
    arrays (``symusic.NoteTickList.numpy()``).
    The velocities are ignored in this method.
    **The notes need to be sorted by time, then pitch, and duration if
    consider_duration is True:**
    ``notes.sort(key=lambda x: (x.start, x.pitch, x.duration))``.

    :param notes: notes to analyse.
    :param consider_duration: if given ``True``, the method will also consider the
        durations of the notes when detecting identical ones. (default: ``False``)
    """
    is_list_of_notes = isinstance(notes, NoteTickList)
    note_soa = notes.numpy() if is_list_of_notes else notes

    keys_to_stack = ["time", "pitch"]
    if consider_duration:
        keys_to_stack.append("duration")
    onset_pitches = np.stack([note_soa[key] for key in keys_to_stack], axis=1)

    successive_val_eq = np.all(onset_pitches[:-1] == onset_pitches[1:], axis=1)
    idx_to_del = np.where(successive_val_eq)[0] + 1

    if is_list_of_notes:
        for idx in reversed(idx_to_del):
            del notes[idx]
    else:
        (mask := np.ones(len(notes["time"]), dtype=bool))[idx_to_del] = False
        for key in notes:
            notes[key] = notes[key][mask]


def fix_offsets_overlapping_notes(notes: NoteTickList) -> None:
    r"""
    Reduce the durations of overlapping notes.

    By applying this method, when a note starts while it is already being played,
    the previous note will end. Before running this method make sure the notes are
    sorted by start then pitch then end values:
    ``notes.sort(key=lambda x: (x.start, x.pitch, x.end))``.

    :param notes: notes to fix.
    """
    for i in range(len(notes) - 1):
        j = i + 1
        while j < len(notes) and notes[j].start <= notes[i].end:
            if notes[i].pitch == notes[j].pitch:
                notes[i].duration = notes[j].start - notes[i].start
                # Breaks here as no other notes with .start before this one
                break
            j += 1


def detect_chords(
    notes: NoteTickList,
    ticks_per_beat: np.array,
    chord_maps: dict[str, Sequence[int]],
    program: int | None = None,
    specify_root_note: bool = True,
    beat_res: int = 4,
    onset_offset: int = 1,
    unknown_chords_num_notes_range: tuple[int, int] | None = None,
    simul_notes_limit: int = 10,
) -> list[Event]:
    r"""
    Detect chords in a list of notes.

    Make sure to sort notes by start time then pitch
    before: ``notes.sort(key=lambda x: (x.start, x.pitch))``.
    **On very large tracks with high note density this method can be slow.**
    If you plan to use it with the Maestro or GiantMIDI datasets, it can take up to
    hundreds of seconds per MIDI depending on your cpu.
    This method works by iterating over each note, find if it played with other notes,
    and if it forms a chord from the chord maps. **It does not consider chord
    inversion.**.

    :param notes: notes to analyse (sorted by starting time, them pitch).
    :param ticks_per_beat: array indicating the number of ticks per beat per
        time signature denominator section. The numbers of ticks per beat depend on the
        time signatures of the MIDI being parsed. The array has a shape ``(N,2)``, for
        ``N`` changes of ticks per beat, and the second dimension representing the end
        tick of each section and the number of ticks per beat respectively.
    :param chord_maps: list of chord maps, to be given as a dictionary where keys are
        chord qualities (e.g. "maj") and values pitch maps as tuples of integers
        (e.g. (0, 4, 7)). You can use ``miditok.constants.CHORD_MAPS`` as an example.
    :param program: program of the track of the notes. Used to specify the program when
        creating the Event object. (default: None)
    :param specify_root_note: the root note of each chord will be specified in
        Events/tokens. Tokens will look as "Chord_C:maj". (default: True)
    :param beat_res: beat resolution, i.e. number of samples per beat (default 4).
    :param onset_offset: maximum offset (in samples) ∈ N separating notes starts to
        consider them starting at the same time / onset (default is 1).
    :param unknown_chords_num_notes_range: range of number of notes to represent unknown
        chords. If you want to represent chords that does not match any combination in
        ``chord_maps``, use this argument. Leave ``None`` to not represent unknown
        chords. (default: None)
    :param simul_notes_limit: number of simultaneous notes being processed when looking
        for a chord this parameter allows to speed up the chord detection, and must be
        >= 5 (default 10).
    :return: the detected chords as Event objects.
    """
    if simul_notes_limit < 5:
        simul_notes_limit = 5
        warnings.warn(
            "`simul_notes_limit` must be >= 5, chords can be made up to 5 notes."
            "Setting it to 5.",
            stacklevel=2,
        )
    notes = np.asarray(
        [(note.pitch, int(note.start), int(note.end)) for note in notes]
    )  # (N,3)

    tpb_idx = 0
    tpb_half = ticks_per_beat[tpb_idx, 1] // 2
    onset_offset_tick = ticks_per_beat[tpb_idx, 1] * onset_offset / beat_res

    count = 0
    previous_tick = -1
    chords = []
    while count < len(notes):
        # Checks we moved in time after last step, otherwise discard this tick
        if notes[count, 1] == previous_tick:
            count += 1
            continue

        # Update the time offset the time signature denom/ticks per beat changed
        # `while` as there might not be any note in the next section
        while notes[count, 1] >= ticks_per_beat[tpb_idx, 0]:
            tpb_idx += 1
            tpb_half = ticks_per_beat[tpb_idx, 1] // 2
            onset_offset_tick = ticks_per_beat[tpb_idx, 1] * onset_offset / beat_res

        # Gathers the notes around the same time step
        onset_notes = notes[count : count + simul_notes_limit]  # reduces the scope
        onset_notes = onset_notes[
            np.where(onset_notes[:, 1] <= onset_notes[0, 1] + onset_offset_tick)
        ]

        # If it is ambiguous, e.g. the notes lengths are too different
        if np.any(np.abs(onset_notes[:, 2] - onset_notes[0, 2]) > tpb_half):
            count += len(onset_notes)
            continue

        # Selects the possible chords notes
        if notes[count, 2] - notes[count, 1] <= tpb_half:
            onset_notes = onset_notes[np.where(onset_notes[:, 1] == onset_notes[0, 1])]
        chord = onset_notes[np.where(onset_notes[:, 2] - onset_notes[0, 2] <= tpb_half)]

        # Creates the "chord map" and see if it has a "known" quality, append a chord
        # event if it is valid
        chord_map = tuple(chord[:, 0] - chord[0, 0])
        if (
            3 <= len(chord_map) <= 5 and chord_map[-1] <= 24
        ):  # max interval between the root and highest degree
            chord_quality = f"{UNKNOWN_CHORD_PREFIX}{len(chord)}"
            is_unknown_chord = True
            for quality, known_chord in chord_maps.items():
                if known_chord == chord_map:
                    chord_quality = quality
                    is_unknown_chord = False
                    break

            # We found a chord quality, or we specify unknown chords
            if unknown_chords_num_notes_range is not None or not is_unknown_chord:
                if specify_root_note:
                    chord_quality = (
                        f"{PITCH_CLASSES[notes[count, 0] % 12]}:{chord_quality}"
                    )
                chords.append(
                    Event(
                        type_="Chord",
                        value=chord_quality,
                        time=min(chord[:, 1]),
                        program=program,
                        desc=chord_map,
                    )
                )

        previous_tick = max(onset_notes[:, 1])
        count += len(onset_notes)  # Move to the next notes

    return chords


def merge_tracks_per_class(
    midi: Score,
    classes_to_merge: list[int] | None = None,
    new_program_per_class: dict[int, int] | None = None,
    max_num_of_tracks_per_inst_class: dict[int, int] | None = None,
    valid_programs: list[int] | None = None,
    filter_pitches: bool = True,
) -> None:
    r"""
    Merge ``symusic.Track``\s per MIDI program class.

    Example, a list of tracks / programs ``[0, 3, 8, 10, 11, 24, 25, 44, 47]`` will
    become ``[0, 8, 24, 25, 40]`` if ``classes_to_merge`` is ``[0, 1, 5]``.
    The classes are in ``miditok.constants.INSTRUMENT_CLASSES``.

    **Note:** programs of drum tracks will be set to -1.

    :param midi: MIDI object to merge tracks
    :param classes_to_merge: instrument classes to merge, to give as list of indexes
        (see miditok.constants.INSTRUMENT_CLASSES). Give None to merge nothing, the
        function will still remove non-valid programs/tracks if given. (default:
        ``None``)
    :param new_program_per_class: new program of the final merged tracks, to be given
        per instrument class as a dict ``{class_id: program}``. (default: ``None``)
    :param max_num_of_tracks_per_inst_class: max number of tracks per instrument class,
        if the limit is exceeded for one class only the tracks with the maximum notes
        will be kept, give None for no limit. (default: ``None``)
    :param valid_programs: valid program ids to keep, others will be deleted, give None
        for keep all programs. (default ``None``)
    :param filter_pitches: after merging, will remove notes whose pitches are out the
        recommended range defined by the GM2 specs. (default: ``True``)
    """
    # remove non-valid tracks (instruments)
    if valid_programs is not None:
        i = 0
        while i < len(midi.tracks):
            if midi.tracks[i].is_drum:
                midi.tracks[i].program = 128  # sets program of drums to 128
            if midi.tracks[i].program not in valid_programs:
                del midi.tracks[i]
                if len(midi.tracks) == 0:
                    return
            else:
                i += 1

    # merge tracks of the same instrument classes
    if classes_to_merge is not None:
        midi.tracks.sort(key=lambda trac: trac.program)
        if max_num_of_tracks_per_inst_class is None:
            max_num_of_tracks_per_inst_class = {
                cla: len(midi.tracks) for cla in classes_to_merge
            }  # no limit
        if new_program_per_class is None:
            new_program_per_class = {
                cla: INSTRUMENT_CLASSES[cla]["program_range"].start
                for cla in classes_to_merge
            }
        else:
            for cla, program in new_program_per_class.items():
                if program not in INSTRUMENT_CLASSES[cla]["program_range"]:
                    msg = (
                        f"Error in program value, got {program} for instrument class"
                        f"{cla} ({INSTRUMENT_CLASSES[cla]['name']}), required value in"
                        f"{INSTRUMENT_CLASSES[cla]['program_range']}"
                    )
                    raise ValueError(msg)

        for ci in classes_to_merge:
            idx_to_merge = [
                ti
                for ti in range(len(midi.tracks))
                if midi.tracks[ti].program in INSTRUMENT_CLASSES[ci]["program_range"]
            ]
            if len(idx_to_merge) > 0:
                midi.tracks[idx_to_merge[0]].program = new_program_per_class[ci]
                if len(idx_to_merge) > max_num_of_tracks_per_inst_class[ci]:
                    lengths = [len(midi.tracks[idx].notes) for idx in idx_to_merge]
                    idx_to_merge = np.argsort(lengths)
                    # could also be randomly picked

                # Merges tracks to merge
                midi.tracks[idx_to_merge[0]] = merge_tracks(
                    [midi.tracks[i] for i in idx_to_merge]
                )

                # Removes tracks merged to index idx_to_merge[0]
                new_len = len(midi.tracks) - len(idx_to_merge) + 1
                while len(midi.tracks) > new_len:
                    del midi.tracks[idx_to_merge[0] + 1]

    # filters notes with pitches out of tessitura / recommended pitch range
    if filter_pitches:
        for track in midi.tracks:
            ni = 0
            while ni < len(track.notes):
                if track.is_drum:
                    tessitura = DRUM_PITCH_RANGE
                else:
                    tessitura = MIDI_INSTRUMENTS[track.program]["pitch_range"]
                if track.notes[ni].pitch not in tessitura:
                    del track.notes[ni]
                else:
                    ni += 1


def merge_tracks(
    tracks: list[Track] | TrackTickList | Score, effects: bool = True
) -> Track:
    r"""
    Merge several ``symusic.Track``\s.

    The notes (and optionally effects) will be concatenated and sorted by time.
    All the tracks will be merged into the first ``Track`` of the list.

    :param tracks: list of tracks to merge, or ``symusic.Score`` object.
    :param effects: will also merge effects, i.e. control changes, sustain pedals and
        pitch bends. (default: ``True``)
    :return: the merged track.
    """
    tracks_ = tracks.tracks if isinstance(tracks, Score) else tracks

    # Change name
    tracks_[0].name += "".join([" / " + t.name for t in tracks_[1:]])

    # Gather and sort notes
    notes_sum = []
    for track in tracks_:
        notes_sum += track.notes
    tracks_[0].notes = notes_sum
    tracks_[0].notes.sort()
    if effects:
        # Pedals
        pedals_sum, cc_sum, pb_sum = [], [], []
        for track in tracks_:
            pedals_sum += track.pedals
            cc_sum += track.controls
            pb_sum += track.pitch_bends
        tracks_[0].pedals = pedals_sum
        tracks_[0].pedals.sort()
        # Control changes
        tracks_[0].controls = cc_sum
        # tracks_[0].controls = sum((t.controls for t in tracks_), [])
        tracks_[0].controls.sort()
        # Pitch bends
        tracks_[0].pitch_bends = pb_sum
        tracks_[0].pitch_bends.sort()

    # Keeps only one track
    if isinstance(tracks, Score):
        tracks.tracks = [tracks_[0]]
    else:
        for _ in range(1, len(tracks)):
            del tracks[1]
        tracks[0] = tracks_[0]
    return tracks_[0]


def merge_same_program_tracks(
    tracks: list[Track] | TrackTickList, effects: bool = True
) -> None:
    r"""
    Merge tracks having the same program number.

    :param tracks: list of tracks.
    :param effects: will also merge effects, i.e. control changes, sustain pedals and
        pitch bends. (default: ``True``)
    """
    # Gathers tracks programs and indexes
    tracks_programs = np.array(
        [int(track.program) if not track.is_drum else -1 for track in tracks]
    )

    # Detects duplicated programs
    unique_programs = np.unique(tracks_programs)
    idx_groups = [np.where(tracks_programs == prog)[0] for prog in unique_programs]
    idx_groups = [group for group in idx_groups if len(group) >= 2]
    if len(idx_groups) == 0:
        return

    # Merge tracks and replace the first idx of each group
    idx_to_delete = []
    for idx_group in reversed(idx_groups):
        new_track = merge_tracks([tracks[idx] for idx in idx_group], effects=effects)
        tracks[idx_group[0]] = new_track
        idx_to_delete.extend(idx_group[1:])

    # Delete tracks merged in first idx of each group
    idx_to_delete.sort(reverse=True)
    for idx in idx_to_delete:
        del tracks[idx]


def num_bar_pos(
    seq: Sequence[int], bar_token_id: int, position_tokens_ids: Sequence[int]
) -> tuple[int, int]:
    r"""
    Return the number of bars and the last position of a sequence of tokens.

    This method is compatible with tokenizations representing time with *Bar* and
    *Position* tokens, such as :py:class:`miditok.REMI`.

    :param seq: sequence of tokens
    :param bar_token_id: the bar token value
    :param position_tokens_ids: position tokens values
    :return: the current bar, current position within the bar, current pitches played
        at this position, and if a chord token has been predicted at this position.
    """
    # Current bar
    bar_idx = [i for i, token in enumerate(seq) if token == bar_token_id]
    current_bar = len(bar_idx)
    # Current position value within the bar
    pos_idx = [
        i for i, token in enumerate(seq[bar_idx[-1] :]) if token in position_tokens_ids
    ]
    current_pos = (
        len(pos_idx) - 1
    )  # position value, e.g. from 0 to 15, -1 means a bar with no Pos token following

    return current_bar, current_pos


def np_get_closest(array: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Find the closest values to those of another reference array.

    Taken from: https://stackoverflow.com/a/46184652.

    :param array: reference values array.
    :param values: array to filter.
    :return: the closest values for each position.
    """
    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return array[idxs]


def tempo_qpm_to_mspq(tempo_qpm: int | float | np.ndarray) -> int | float | np.ndarray:
    """
    Convert tempo value(s) in qpm (quarter/minute) to mspq (μs/quarter).

    This method works with either a single qpm value (``int`` or ``float``), or with
    several as a ``numpy.ndarray``.

    :param tempo_qpm: tempo value(s) in qpm to convert.
    :return: array of equivalent values in mspq.
    """
    # quarter / 60sec --> 1μs / quarter
    return 60000000 / tempo_qpm


def miditoolkit_to_symusic(midi: MidiFile) -> Score:
    """
    Convert a ``miditoolkit.MidiFile`` object into a ``symusic.Score``.

    :param midi: ``miditoolkit.MidiFile`` object to convert.
    :return: the `symusic.Score`` equivalent.
    """
    score = Score(midi.ticks_per_beat)

    # MIDI events (except key signature)
    for time_sig in midi.time_signature_changes:
        score.time_signatures.append(
            TimeSignature(time_sig.time, time_sig.numerator, time_sig.denominator)
        )
    for tempo in midi.tempo_changes:
        score.tempos.append(Tempo(tempo.time, tempo.tempo))
    for lyric in midi.lyrics:
        score.lyrics.append(TextMeta(lyric.time, lyric.text))
    for marker in midi.markers:
        score.markers.append(TextMeta(marker.time, marker.text))

    # Track events
    for inst in midi.instruments:
        track = Track(
            name=inst.name,
            program=inst.program,
            is_drum=inst.is_drum,
        )
        for note in inst.notes:
            track.notes.append(
                Note(note.start, note.duration, note.pitch, note.velocity)
            )
        track.notes.sort(key=lambda x: (x.start, x.pitch, x.end, x.velocity))

        for control in inst.control_changes:
            track.controls.append(
                ControlChange(control.time, control.number, control.value)
            )
        track.controls.sort()

        for pb in inst.pitch_bends:
            track.pitch_bends.append(PitchBend(pb.time, pb.pitch))
        track.pitch_bends.sort()

        for pedal in inst.pedals:
            track.pedals.append(Pedal(pedal.start, pedal.duration))
        track.pedals.sort()

        score.tracks.append(track)

    return score


def compute_ticks_per_beat(time_sig_denominator: int, time_division: int) -> int:
    r"""
    Compute the number of ticks in a beat at a given time signature.

    :param time_sig_denominator: time signature denominator.
    :param time_division: MIDI time division in ticks/quarter.
    :return: number of ticks per beat at the given time signature.
    """
    if time_sig_denominator == 4:
        return time_division
    # factor to multiply the time_division depending on the denominator
    # if we have a */8 time sig, one beat is an eighth note so one beat is
    # `time_division * 0.5` ticks.
    time_div_factor = 4 / time_sig_denominator
    return int(time_division * time_div_factor)


def compute_ticks_per_bar(time_sig: TimeSignature, time_division: int) -> int:
    r"""
    Compute the number of ticks in a bar.

    The number of ticks per bar depends on the time signature.

    :param time_sig: time signature object.
    :param time_division: MIDI time division in ticks/quarter.
    :return: MIDI bar resolution, in ticks/bar
    """
    return int(
        compute_ticks_per_beat(time_sig.denominator, time_division) * time_sig.numerator
    )


def get_bars_ticks(midi: Score) -> list[int]:
    """
    Compute the ticks of the bars of a MIDI.

    **Note:** When encountering multiple time signature messages at a same tick, we
    this method will automatically consider the last one (coming in the list). Other
    software can proceed differently. Logic Pro, for example, uses the first one.
    I haven't found documentation or recommendations for this specific situation. It
    might be better to use the first one and discard the others.

    :param midi: MIDI to analyze.
    :return: list of ticks for each bar.
    """
    max_tick = midi.end()
    bars_ticks = []
    time_sigs = copy(midi.time_signatures)
    # Mock the last one to cover the last section in the loop below
    if time_sigs[-1].time != max_tick:
        time_sigs.append(TimeSignature(max_tick, *TIME_SIGNATURE))

    # Section from tick 0 to first time sig is 4/4 if first time sig time is not 0
    if time_sigs[0].time == 0:
        current_time_sig = time_sigs[0]
    else:
        current_time_sig = TimeSignature(0, *TIME_SIGNATURE)

    # Compute bars, one time signature portion at a time
    for time_signature in time_sigs:
        ticks_per_bar = compute_ticks_per_bar(current_time_sig, midi.ticks_per_quarter)
        ticks_diff = time_signature.time - current_time_sig.time
        num_bars = ceil(ticks_diff / ticks_per_bar)
        bars_ticks += [
            current_time_sig.time + ticks_per_bar * i for i in range(num_bars)
        ]
        current_time_sig = time_signature

    return bars_ticks


def get_midi_ticks_per_beat(midi: Score) -> np.ndarray:
    """
    Compute the portions of numbers of ticks in a beat in a MIDI.

    The method returns a numpy array of shape ``(N,2)``, for N ticks-per-beat changes,
    and the second dimension corresponding to the ending tick and the number of ticks
    per beat of the portion.

    :param midi: MIDI to analyze.
    :return: ticks per beat values as a numpy array.
    """
    ticks_per_beat = [
        [
            midi.time_signatures[tsi + 1].time,
            compute_ticks_per_beat(
                midi.time_signatures[tsi].denominator, midi.ticks_per_quarter
            ),
        ]
        for tsi in range(len(midi.time_signatures) - 1)
    ]

    # Handles the last one up to the max tick of the MIDI
    ticks_per_beat.append(
        [
            midi.end() + 1,
            compute_ticks_per_beat(
                midi.time_signatures[-1].denominator, midi.ticks_per_quarter
            ),
        ]
    )

    # Remove equal successive ones
    for i in range(len(ticks_per_beat) - 1, 0, -1):
        if ticks_per_beat[i][1] == ticks_per_beat[i - 1][1]:
            ticks_per_beat[i - 1][0] = ticks_per_beat[i][0]
            del ticks_per_beat[i]

    return np.array(ticks_per_beat)
