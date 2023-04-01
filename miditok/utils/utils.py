"""Useful methods

"""

from typing import List, Tuple, Sequence, Dict, Union, Any
from collections import Counter

from miditoolkit import MidiFile, Note, Instrument
import numpy as np

from miditok.classes import Event
from miditok.constants import (
    INSTRUMENT_CLASSES,
    MIDI_INSTRUMENTS,
    PITCH_CLASSES,
    UNKNOWN_CHORD_PREFIX,
)


def convert_ids_tensors_to_list(ids: Any):
    """Convert a PyTorch, Tensorflow Tensor or numpy array to a list of integers.
    This method can be used seamlessly with Jax too.

    :param ids: ids sequence to convert.
    :return: the input, as a list of integers
    """
    # Convert tokens to list if necessary
    if not isinstance(ids, list):
        if type(ids).__name__ in ["Tensor", "EagerTensor"]:
            ids = ids.numpy()
        if not isinstance(ids, np.ndarray):
            raise TypeError(
                "The tokens must be given as a list of integers, np.ndarray, or PyTorch / Tensorflow tensor"
            )
        ids = ids.astype(int).tolist()

    return ids


def get_midi_programs(midi: MidiFile) -> List[Tuple[int, bool]]:
    r"""Returns the list of programs of the tracks of a MIDI, deeping the
    same order. It returns it as a list of tuples (program, is_drum).

    :param midi: the MIDI object to extract tracks programs
    :return: the list of track programs, as a list of tuples (program, is_drum)
    """
    return [(int(track.program), track.is_drum) for track in midi.instruments]


def remove_duplicated_notes(notes: List[Note]):
    r"""Removes (inplace) possible duplicated notes, i.e. with the same pitch, starting and ending times.
    Before running this function make sure the notes has been sorted by start then pitch then end values:
    notes.sort(key=lambda x: (x.start, x.pitch, x.end))

    :param notes: notes to analyse
    """
    for i in range(len(notes) - 1, 0, -1):  # removing possible duplicated notes
        if (
            notes[i].pitch == notes[i - 1].pitch
            and notes[i].start == notes[i - 1].start
            and notes[i].end >= notes[i - 1].end
        ):
            del notes[i]


def detect_chords(
    notes: Sequence[Note],
    time_division: int,
    chord_maps: Dict[str, Sequence[int]],
    specify_root_note: bool = True,
    beat_res: int = 4,
    onset_offset: int = 1,
    unknown_chords_nb_notes_range: Union[bool, Tuple[int]] = False,
    simul_notes_limit: int = 10,
) -> List[Event]:
    r"""Chord detection method. Make sure to sort notes by start time then pitch before:
    ``notes.sort(key=lambda x: (x.start, x.pitch))``.
    **On very large tracks with high note density this method can slow down processes.**
    If you plan to use it with the Maestro or GiantMIDI datasets, it can take up to
    hundreds of seconds per MIDI depending on your cpu.
    This method works by iterating over each note, find if it played with other notes, and if it
    forms a chord from the chord maps. **It does not consider chord inversion.**

    :param notes: notes to analyse (sorted by starting time, them pitch)
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param chord_maps: list of chord maps, to be given as a dictionary where keys are chord qualities
                    (e.g. "maj") and values pitch maps as tuples of integers (e.g. (0, 4, 7)).
                    You can use or take as an example ``miditok.constants.CHORD_MAPS``.
    :param specify_root_note: the root note of each chord will be specified in Events / tokens (default: True).
    :param beat_res: beat resolution, i.e. nb of samples per beat (default 4).
    :param onset_offset: maximum offset (in samples) ∈ N separating notes starts to consider them
                    starting at the same time / onset (default is 1).
    :param unknown_chords_nb_notes_range: will detect only chords recognized in the chord maps. If set to False,
                    non recognized chords of *n* notes will give a *Chord_n* token (default False).
    :param simul_notes_limit: nb of simultaneous notes being processed when looking for a chord
            this parameter allows to speed up the chord detection, and must be >= 5 (default 10).
    :return: the detected chords as Event objects.
    """
    assert (
        simul_notes_limit >= 5
    ), "simul_notes_limit must be higher than 5, chords can be made up to 5 notes"
    tuples = []
    for note in notes:
        tuples.append((note.pitch, int(note.start), int(note.end)))
    notes = np.asarray(tuples)  # (N,3)

    time_div_half = time_division // 2
    onset_offset = time_division * onset_offset / beat_res

    count = 0
    previous_tick = -1
    chords = []
    while count < len(notes):
        # Checks we moved in time after last step, otherwise discard this tick
        if notes[count, 1] == previous_tick:
            count += 1
            continue

        # Gathers the notes around the same time step
        onset_notes = notes[count : count + simul_notes_limit]  # reduces the scope
        onset_notes = onset_notes[
            np.where(onset_notes[:, 1] <= onset_notes[0, 1] + onset_offset)
        ]

        # If it is ambiguous, e.g. the notes lengths are too different
        if np.any(np.abs(onset_notes[:, 2] - onset_notes[0, 2]) > time_div_half):
            count += len(onset_notes)
            continue

        # Selects the possible chords notes
        if notes[count, 2] - notes[count, 1] <= time_div_half:
            onset_notes = onset_notes[np.where(onset_notes[:, 1] == onset_notes[0, 1])]
        chord = onset_notes[
            np.where(onset_notes[:, 2] - onset_notes[0, 2] <= time_div_half)
        ]

        # Creates the "chord map" and see if it has a "known" quality, append a chord event if it is valid
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
            if not (unknown_chords_nb_notes_range is False and is_unknown_chord):
                if specify_root_note:
                    chord_quality = (
                        f"{PITCH_CLASSES[notes[count, 0] % 12]}:{chord_quality}"
                    )
                chords.append(
                    Event(
                        type="Chord",
                        value=chord_quality,
                        time=min(chord[:, 1]),
                        desc=chord_map,
                    )
                )

        previous_tick = max(onset_notes[:, 1])
        count += len(onset_notes)  # Move to the next notes

    return chords


def merge_tracks_per_class(
    midi: MidiFile,
    classes_to_merge: List[int] = None,
    new_program_per_class: Dict[int, int] = None,
    max_nb_of_tracks_per_inst_class: Dict[int, int] = None,
    valid_programs: List[int] = None,
    filter_pitches: bool = True,
):
    r"""Merges per instrument class the tracks which are in the class in classes_to_merge.
    Example, a list of tracks / programs [0, 3, 8, 10, 11, 24, 25, 44, 47] will become [0, 8, 24, 25, 40] if
    classes_to_merge is [0, 1, 5].
    The classes are in ``miditok.constants.INSTRUMENT_CLASSES``.

    **Note:** programs of drum tracks will be set to -1.

    :param midi: MIDI object to merge tracks
    :param classes_to_merge: instrument classes to merge, to give as list of indexes
            (see miditok.constants.INSTRUMENT_CLASSES). Give None to merge nothing,
            the function will still remove non-valid programs / tracks if given (default: None)
    :param new_program_per_class: new program of the final merged tracks, to be given per
            instrument class as a dict {class_id: program}
    :param max_nb_of_tracks_per_inst_class: max number of tracks per instrument class,
            if the limit is exceeded for one class only the tracks with the maximum notes
            will be kept, give None for no limit (default: None)
    :param valid_programs: valid program ids to keep, others will be deleted, give None
            for keep all programs (default None)
    :param filter_pitches: after merging, will remove notes whose pitches are out the recommended
            range defined by the GM2 specs (default: True)
    :return: True if the MIDI is valid, else False
    """
    # remove non-valid tracks (instruments)
    if valid_programs is not None:
        i = 0
        while i < len(midi.instruments):
            if midi.instruments[i].is_drum:
                midi.instruments[i].program = -1  # sets program of drums to -1
            if midi.instruments[i].program not in valid_programs:
                del midi.instruments[i]
                if len(midi.instruments) == 0:
                    return False
            else:
                i += 1

    # merge tracks of the same instrument classes
    if classes_to_merge is not None:
        midi.instruments.sort(key=lambda trac: trac.program)
        if max_nb_of_tracks_per_inst_class is None:
            max_nb_of_tracks_per_inst_class = {
                cla: len(midi.instruments) for cla in classes_to_merge
            }  # no limit
        if new_program_per_class is None:
            new_program_per_class = {
                cla: INSTRUMENT_CLASSES[cla]["program_range"].start
                for cla in classes_to_merge
            }
        else:
            for cla, program in new_program_per_class.items():
                if program not in INSTRUMENT_CLASSES[cla]["program_range"]:
                    raise ValueError(
                        f"Error in program value, got {program} for instrument class {cla} "
                        f'({INSTRUMENT_CLASSES[cla]["name"]}), required value in '
                        f'{INSTRUMENT_CLASSES[cla]["program_range"]}'
                    )

        for ci in classes_to_merge:
            idx_to_merge = [
                ti
                for ti in range(len(midi.instruments))
                if midi.instruments[ti].program
                in INSTRUMENT_CLASSES[ci]["program_range"]
            ]
            if len(idx_to_merge) > 0:
                midi.instruments[idx_to_merge[0]].program = new_program_per_class[ci]
                if len(idx_to_merge) > max_nb_of_tracks_per_inst_class[ci]:
                    lengths = [len(midi.instruments[idx].notes) for idx in idx_to_merge]
                    idx_to_merge = np.argsort(lengths)
                    # could also be randomly picked

                # Merges tracks to merge
                midi.instruments[idx_to_merge[0]] = merge_tracks(
                    [midi.instruments[i] for i in idx_to_merge]
                )

                # Removes tracks merged to index idx_to_merge[0]
                new_len = len(midi.instruments) - len(idx_to_merge) + 1
                while len(midi.instruments) > new_len:
                    del midi.instruments[idx_to_merge[0] + 1]

    # filters notes with pitches out of tessitura / recommended pitch range
    if filter_pitches:
        for track in midi.instruments:
            ni = 0
            while ni < len(track.notes):
                if (
                    track.notes[ni].pitch
                    not in MIDI_INSTRUMENTS[track.program]["pitch_range"]
                ):
                    del track.notes[ni]
                else:
                    ni += 1


def merge_tracks(
    tracks: Union[List[Instrument], MidiFile], effects: bool = False
) -> Instrument:
    r"""Merge several miditoolkit Instrument objects, from a list of Instruments or a MidiFile object.
    All the tracks will be merged into the first Instrument object (notes concatenated and sorted),
    beware of giving tracks with the same program (no assessment is performed).
    The other tracks will be deleted.

    :param tracks: list of tracks to merge, or MidiFile object
    :param effects: will also merge effects, i.e. control changes, sustain pedals and pitch bends
    :return: the merged track
    """
    if isinstance(tracks, MidiFile):
        tracks_ = tracks.instruments
    else:
        tracks_ = tracks

    # Change name
    tracks_[0].name += "".join([" / " + t.name for t in tracks_[1:]])

    # Gather and sort notes
    tracks_[0].notes = sum((t.notes for t in tracks_), [])
    tracks_[0].notes.sort(key=lambda note: note.start)
    if effects:
        # Pedals
        tracks_[0].pedals = sum((t.pedals for t in tracks_), [])
        tracks_[0].pedals.sort(key=lambda pedal: pedal.start)
        # Control changes
        tracks_[0].control_changes = sum((t.control_changes for t in tracks_), [])
        tracks_[0].control_changes.sort(key=lambda control_change: control_change.time)
        # Pitch bends
        tracks_[0].pitch_bends = sum((t.pitch_bends for t in tracks_), [])
        tracks_[0].pitch_bends.sort(key=lambda pitch_bend: pitch_bend.start)

    # Keeps only one track
    if isinstance(tracks, MidiFile):
        tracks.instruments = [tracks_[0]]
    else:
        for _ in range(1, len(tracks)):
            del tracks[1]
        tracks[0] = tracks_[0]
    return tracks_[0]


def merge_same_program_tracks(tracks: List[Instrument]):
    r"""Takes a list of tracks and merge the ones with the same programs.
    NOTE: Control change messages are not considered

    :param tracks: list of tracks
    """
    # Gathers tracks programs and indexes
    tracks_programs = [
        int(track.program) if not track.is_drum else -1 for track in tracks
    ]

    # Detects duplicated programs
    duplicated_programs = [k for k, v in Counter(tracks_programs).items() if v > 1]

    # Merges duplicated tracks
    for program in duplicated_programs:
        idx = [
            i
            for i in range(len(tracks))
            if (
                tracks[i].is_drum
                if program == -1
                else tracks[i].program == program and not tracks[i].is_drum
            )
        ]
        tracks[idx[0]].name += "".join([" / " + tracks[i].name for i in idx[1:]])
        tracks[idx[0]].notes = sum((tracks[i].notes for i in idx), [])
        tracks[idx[0]].notes.sort(key=lambda note: (note.start, note.pitch))
        for i in list(reversed(idx[1:])):
            del tracks[i]


def nb_bar_pos(
    seq: Sequence[int], bar_token: int, position_tokens: Sequence[int]
) -> Tuple[int, int]:
    r"""Returns the number of bars and the last position of a sequence of tokens. This method
    is compatible with tokenizations representing time with *Bar* and *Position* tokens, such as
    :py:class:`miditok.REMI`.

    :param seq: sequence of tokens
    :param bar_token: the bar token value
    :param position_tokens: position tokens values
    :return: the current bar, current position within the bar, current pitches played at this position,
            and if a chord token has been predicted at this position
    """
    # Current bar
    bar_idx = [i for i, token in enumerate(seq) if token == bar_token]
    current_bar = len(bar_idx)
    # Current position value within the bar
    pos_idx = [
        i for i, token in enumerate(seq[bar_idx[-1] :]) if token in position_tokens
    ]
    current_pos = (
        len(pos_idx) - 1
    )  # position value, e.g. from 0 to 15, -1 means a bar with no Pos token following

    return current_bar, current_pos
