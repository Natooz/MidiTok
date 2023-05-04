"""Data augmentation methods

"""

from typing import List, Tuple, Union, Dict
from pathlib import Path
from copy import deepcopy
from warnings import warn
import json

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile


def data_augmentation_dataset(
    data_path: Union[Path, str],
    tokenizer=None,
    nb_octave_offset: int = None,
    nb_vel_offset: int = None,
    nb_dur_offset: int = None,
    octave_directions: Tuple[bool, bool] = (True, True),
    vel_directions: Tuple[bool, bool] = (True, True),
    dur_directions: Tuple[bool, bool] = (True, True),
    all_offset_combinations: bool = False,
    out_path: Union[Path, str] = None,
    copy_original_in_new_location: bool = True,
):
    r"""Perform data augmentation on a whole dataset, on the pitch dimension.
    Drum tracks are not augmented.
    The new created files have names in two parts, separated with a '§' character.
    Make sure your files do not have '§' in their names if you intend to reuse the information of the
    second part in some script.

    :param data_path: root path to the folder containing tokenized json files.
    :param tokenizer: tokenizer, needs to have 'Pitch' or 'NoteOn' tokens. Has to be given
            if performing augmentation on tokens (default: None).
    :param nb_octave_offset: number of pitch octaves offset to perform data augmentation.
    :param nb_vel_offset: number of velocity values
    :param nb_dur_offset: number of pitch octaves offset to perform data augmentation.
    :param octave_directions: directions to shift the pitch augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param vel_directions: directions to shift the velocity augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param dur_directions: directions to shift the duration augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param all_offset_combinations: will perform data augmentation on all the possible
            combinations of offsets. If set to False, will perform data augmentation
            only based on the original sample.
    :param out_path: output path to save the augmented files. Original (non-augmented) MIDIs will be
            saved to this location. If none is given, they will be saved in the same location an the
            data_path. (default: None)
    :param copy_original_in_new_location: if given True, the orinal (non-augmented) MIDIs will be saved
            in the out_path location too. (default: True)
    """
    if out_path is None:
        out_path = data_path
    else:
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
    as_tokens = True
    files_paths = list(Path(data_path).glob("**/*.json"))
    if len(files_paths) == 0:
        files_paths = list(Path(data_path).glob("**/*.mid"))
        as_tokens = False

    nb_augmentations, nb_tracks_augmented = 0, 0
    for file_path in tqdm(files_paths, desc="Performing data augmentation"):
        if as_tokens:
            with open(file_path) as json_file:
                file = json.load(json_file)
                ids, programs = file["ids"], file["programs"]

            if tokenizer.unique_track:
                ids = [ids]

            # Perform data augmentation for each track
            offsets = get_offsets(
                tokenizer,
                nb_octave_offset,
                nb_vel_offset,
                nb_dur_offset,
                octave_directions,
                vel_directions,
                dur_directions,
                ids=ids,
            )
            augmented_tokens: Dict[
                Tuple[int, int, int], List[Union[int, List[int]]]
            ] = {}
            for track, (_, is_drum) in zip(ids, programs):
                if is_drum:  # we dont augment drums
                    continue
                corrected_offsets = deepcopy(offsets)
                vel_dim = int(128 / len(tokenizer.velocities))
                corrected_offsets[1] = [
                    int(off / vel_dim) for off in corrected_offsets[1]
                ]
                aug = data_augmentation_tokens(
                    np.array(track),
                    tokenizer,
                    *corrected_offsets,
                    all_offset_combinations=all_offset_combinations,
                )
                if len(aug) == 0:
                    continue
                for aug_offsets, seq in aug:
                    if tokenizer.unique_track:
                        augmented_tokens[aug_offsets] = seq
                        continue
                    try:
                        augmented_tokens[aug_offsets].append(seq)
                    except KeyError:
                        augmented_tokens[aug_offsets] = [seq]
            for i, (track, (_, is_drum)) in enumerate(
                zip(ids, programs)
            ):  # adding drums to all already augmented
                if is_drum:
                    for aug_offsets in augmented_tokens:
                        augmented_tokens[aug_offsets].insert(i, track)

            # Save augmented tracks as json
            for aug_offsets, tracks_seq in augmented_tokens.items():
                if len(tracks_seq) == 0:
                    continue
                suffix = "§" + "_".join(
                    [
                        f"{t}{offset}"
                        for t, offset in zip(["p", "v", "d"], aug_offsets)
                        if offset != 0
                    ]
                )
                saving_path = (
                    file_path.parent if out_path is None else out_path
                ) / f"{file_path.stem}{suffix}.json"
                tokenizer.save_tokens(tracks_seq, saving_path, programs)
                nb_augmentations += 1
                nb_tracks_augmented += len(tracks_seq)
            if copy_original_in_new_location and out_path is not None:
                tokenizer.save_tokens(
                    ids, out_path / f"{file_path.stem}.json", programs
                )

        else:  # as midi
            try:
                midi = MidiFile(file_path)
            except (
                Exception
            ):  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            offsets = get_offsets(
                tokenizer,
                nb_octave_offset,
                nb_vel_offset,
                nb_dur_offset,
                octave_directions,
                vel_directions,
                dur_directions,
                midi=midi,
            )
            augmented_midis = data_augmentation_midi(
                midi,
                tokenizer,
                *offsets,
                all_offset_combinations=all_offset_combinations,
            )
            for aug_offsets, aug_midi in augmented_midis:
                if len(aug_midi.instruments) == 0:
                    continue
                suffix = "§" + "_".join(
                    [
                        f"{t}{offset}"
                        for t, offset in zip(["p", "v", "d"], aug_offsets)
                        if offset != 0
                    ]
                )
                saving_path = (
                    file_path.parent if out_path is None else out_path
                ) / f"{file_path.stem}{suffix}.mid"
                aug_midi.dump(saving_path)
                nb_augmentations += 1
                nb_tracks_augmented += len(aug_midi.instruments)
            if (
                copy_original_in_new_location and out_path is not None
            ):  # copy original midi
                midi.dump(out_path / f"{file_path.stem}.mid")

    # Saves data augmentation report, json encoded with txt extension to not mess with others json files
    with open(data_path / "data_augmentation.txt", "w") as outfile:
        json.dump(
            {
                "nb_tracks_augmented": nb_tracks_augmented,
                "nb_files_before": len(files_paths),
                "nb_files_after": len(files_paths) + nb_augmentations,
            },
            outfile,
        )


def get_offsets(
    tokenizer=None,
    nb_octave_offset: int = None,
    nb_vel_offset: int = None,
    nb_dur_offset: int = None,
    octave_directions: Tuple[bool, bool] = (True, True),
    vel_directions: Tuple[bool, bool] = (True, True),
    dur_directions: Tuple[bool, bool] = (True, True),
    midi: MidiFile = None,
    ids: List[Union[int, List[int]]] = None,
) -> List[List[int]]:
    r"""Build the offsets in absolute value for data augmentation.
    TODO some sort of limit for velocity and duration values (min / max as for octaves)

    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_octave_offset: number of pitch octaves offset to perform data augmentation.
    :param nb_vel_offset: number of velocity values
    :param nb_dur_offset: number of pitch octaves offset to perform data augmentation.
    :param octave_directions: directions to shift the pitch augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param vel_directions: directions to shift the velocity augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param dur_directions: directions to shift the duration augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param midi: midi object to augment (default: None)
    :param ids: token ids as a list of tracks (default: None)
    :return: the offsets of pitch, velocity and duration features, in "absolute" value
    """
    offsets = []

    if nb_octave_offset is not None:
        # Get the maximum and lowest pitch in original track
        all_pitches = []
        if midi is not None:
            for track in midi.instruments:
                if not track.is_drum:
                    all_pitches += [note.pitch for note in track.notes]
            max_pitch, min_pitch = max(all_pitches), min(all_pitches)
        else:
            pitch_voc_idx = (
                tokenizer.vocab_types_idx["Pitch"] if tokenizer.is_multi_voc else None
            )
            ids_pitch = []
            if not tokenizer.is_multi_voc:
                pitch_ids_vocab = np.array(
                    tokenizer.token_ids_of_type("Pitch")
                    + tokenizer.token_ids_of_type("NoteOn")
                )
                for track_ids in ids:
                    tt_arr = np.array(track_ids)
                    ids_pitch.append(tt_arr[np.isin(tt_arr, pitch_ids_vocab)])
                max_pitch = tokenizer[int(np.max(np.concatenate(ids_pitch)))].split(
                    "_"
                )[1]
                min_pitch = tokenizer[int(np.min(np.concatenate(ids_pitch)))].split(
                    "_"
                )[1]
            else:
                pitch_ids_vocab = np.array(
                    tokenizer.token_ids_of_type("Pitch", pitch_voc_idx)
                )
                for track_ids in ids:
                    tt_arr = np.array(track_ids)[:, pitch_voc_idx]
                    ids_pitch.append(tt_arr[np.isin(tt_arr, pitch_ids_vocab)])
                max_pitch = tokenizer[
                    pitch_voc_idx, int(np.max(np.concatenate(ids_pitch)))
                ].split("_")[1]
                min_pitch = tokenizer[
                    pitch_voc_idx, int(np.min(np.concatenate(ids_pitch)))
                ].split("_")[1]
        offset_up = min(
            nb_octave_offset, (tokenizer.pitch_range.stop - 1 - int(max_pitch)) // 12
        )
        offset_down = min(
            nb_octave_offset, (int(min_pitch) - tokenizer.pitch_range.start) // 12
        )

        off = []
        if octave_directions[0]:
            off += list(range(12, offset_up * 12 + 1, 12))
        if octave_directions[1]:
            off += list(range(-offset_down * 12, 0, 12))
        offsets.append(off)

    if nb_vel_offset is not None:
        vel_dim = int(128 / len(tokenizer.velocities))
        off = []
        if vel_directions[0]:
            off += list(range(vel_dim, nb_vel_offset * vel_dim + 1, vel_dim))
        if vel_directions[1]:
            off += list(range(-nb_vel_offset * vel_dim, 0, vel_dim))
        offsets.append(off)

    if nb_dur_offset is not None:
        off = []
        if dur_directions[0]:
            off += list(range(1, nb_dur_offset + 1))
        if dur_directions[1]:
            off += list(range(-nb_dur_offset, 0))
        offsets.append(off)

    return offsets


def data_augmentation_midi(
    midi: MidiFile,
    tokenizer,
    pitch_offsets: List[int] = None,
    velocity_offsets: List[int] = None,
    duration_offsets: List[int] = None,
    all_offset_combinations: bool = False,
) -> List[Tuple[Tuple[int, int, int], MidiFile]]:
    r"""Perform data augmentation on a MIDI object.
    Drum tracks are not augmented, but copied as original in augmented MIDIs.

    :param midi: midi object to augment
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param pitch_offsets: list of pitch offsets for augmentation.
    :param velocity_offsets: list of velocity offsets for augmentation.
    :param duration_offsets: list of duration offsets for augmentation.
    :param all_offset_combinations: will perform data augmentation on all the possible
            combinations of offsets. If set to False, will perform data augmentation
            only based on the original sample.
    :return: augmented MIDI objects.
    """
    augmented = []

    # Pitch augmentation
    if pitch_offsets is not None:
        for offset in pitch_offsets:
            midi_augmented = deepcopy(midi)
            for track in midi_augmented.instruments:
                if not track.is_drum:
                    for note in track.notes:
                        note.pitch += offset
            augmented.append(((offset, 0, 0), midi_augmented))

    # Velocity augmentation
    if velocity_offsets is not None:

        def augment_vel(
            midi_: MidiFile, offsets_: Tuple[int, int, int]
        ) -> List[Tuple[Tuple[int, int, int], MidiFile]]:
            aug_ = []
            for offset_ in velocity_offsets:
                midi_aug_ = deepcopy(midi_)
                for track_ in midi_aug_.instruments:
                    for note_ in track_.notes:
                        if offset_ < 0:
                            note_.velocity = max(
                                min(tokenizer.velocities), note_.velocity + offset_
                            )
                        else:
                            note_.velocity = min(
                                max(tokenizer.velocities), note_.velocity + offset_
                            )
                aug_.append(((offsets_[0], offset_, offsets_[2]), midi_aug_))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, midi_aug = augmented[i]
                augmented += augment_vel(
                    midi_aug, offsets
                )  # for already augmented midis
        augmented += augment_vel(midi, (0, 0, 0))  # for original midi

    # TODO Duration augmentation
    """if duration_offsets is not None:
        tokenizer.durations_ticks[midi.ticks_per_beat] = np.array([(beat * res + pos) * midi.ticks_per_beat // res
                                                                           for beat, pos, res in tokenizer.durations])
        def augment_dur(midi_: MidiFile, offsets_: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], MidiFile]]:
            aug_ = []
            dur_bins = tokenizer.durations_ticks[tokenizer.current_midi_metadata['time_division']]

            for offset_ in dur_offsets:
                midi_aug_ = deepcopy(midi_)
                for track_ in midi_aug_.instruments:
                    for note_ in track_.notes:
                        duration = note.end - note.start
                        index = np.argmin(np.abs(dur_bins - duration))

                        note_.end += offset_  # TODO multiply token val
                aug_.append(((offsets_[0], offsets_[1], offset_), midi_aug_))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, midi_aug = augmented[i]
                augmented += augment_dur(midi_aug, offsets)  # for already augmented midis
        augmented += augment_dur(midi, (0, 0, 0))  # for original midi"""

    return augmented


def data_augmentation_tokens(
    tokens: Union[np.ndarray, List[int]],
    tokenizer,
    pitch_offsets: List[int] = None,
    velocity_offsets: List[int] = None,
    duration_offsets: List[int] = None,
    all_offset_combinations: bool = False,
) -> List[Tuple[Tuple[int, int, int], List[int]]]:
    r"""Perform data augmentation on a sequence of tokens, on the pitch dimension.
    NOTE: token sequences with BPE will be decoded during the augmentation, this might take some time.
    NOTE 2: the tokenizer must have a vocabulary in which the pitch values increase with the token index,
    e.g. Pitch_48: token 64, Pitch_49: token65 ...
    The same goes for durations.
    MIDILike is not compatible with data augmentation on durations.
    MuMIDI is not compatible at all.

    :param tokens: tokens to perform data augmentation on.
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param pitch_offsets: list of pitch offsets for augmentation.
    :param velocity_offsets: list of velocity offsets for augmentation.
    :param duration_offsets: list of duration offsets for augmentation.
    :param all_offset_combinations: will perform data augmentation on all the possible
            combinations of offsets. If set to False, will perform data augmentation
            only based on the original sample.
    :return: the several data augmentations that have been performed
    """
    augmented = []

    # Decode BPE
    bpe_decoded = False
    if tokenizer.has_bpe:
        tokens = tokenizer.decode_bpe(
            tokens.tolist() if isinstance(tokens, np.ndarray) else tokens
        )
        bpe_decoded = True

    # Converts to np array if necessary
    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    if pitch_offsets is not None:
        # Get the maximum and lowest pitch in original track
        pitch_voc_idx = (
            tokenizer.vocab_types_idx["Pitch"] if tokenizer.is_multi_voc else None
        )
        note_off_tokens = []
        if not tokenizer.is_multi_voc:
            pitch_tokens = np.array(
                tokenizer.token_ids_of_type("Pitch")
                + tokenizer.token_ids_of_type("NoteOn")
            )
            note_off_tokens = np.array(tokenizer.token_ids_of_type("NoteOff"))
            mask_pitch = np.isin(tokens, pitch_tokens)
        else:
            pitch_tokens = np.array(tokenizer.token_ids_of_type("Pitch", pitch_voc_idx))
            mask_pitch = np.full_like(tokens, 0, dtype=np.bool_)
            mask_pitch[:, pitch_voc_idx] = np.isin(
                tokens[:, pitch_voc_idx], pitch_tokens
            )

        # Perform augmentation on pitch
        for offset in pitch_offsets:
            seq = tokens.copy()
            seq[mask_pitch] += offset
            if len(note_off_tokens) > 0:  # for MIDIlike
                seq[np.isin(seq, note_off_tokens)] += offset
            augmented.append(((offset, 0, 0), seq))

    # Velocity augmentation
    if velocity_offsets is not None:
        vel_voc_idx = (
            tokenizer.vocab_types_idx["Velocity"] if tokenizer.is_multi_voc else None
        )
        if not tokenizer.is_multi_voc:
            vel_tokens = np.array(tokenizer.token_ids_of_type("Velocity"))
        else:
            vel_tokens = np.array(tokenizer.token_ids_of_type("Velocity", vel_voc_idx))

        def augment_vel(
            seq_: np.ndarray, offsets_: Tuple[int, int, int]
        ) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
            if not tokenizer.is_multi_voc:
                mask = np.isin(seq_, vel_tokens)
            else:
                mask = np.full_like(seq_, 0, dtype=np.bool_)
                mask[:, vel_voc_idx] = np.isin(seq_[:, vel_voc_idx], vel_tokens)

            aug_ = []
            for offset_ in velocity_offsets:
                aug_seq = seq_.copy()
                aug_seq[mask] += offset_
                aug_seq[mask] = np.clip(aug_seq[mask], vel_tokens[0], vel_tokens[-1])
                aug_.append(((offsets_[0], offset_, offsets_[2]), aug_seq))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, seq_aug = augmented[i]
                augmented += augment_vel(
                    seq_aug, offsets
                )  # for already augmented midis
        augmented += augment_vel(tokens, (0, 0, 0))  # for original midi

    # Duration augmentation
    if duration_offsets is not None and type(tokenizer).__name__ == "MIDILike":
        warn(
            f"{type(tokenizer).__name__} is not compatible with data augmentation on duration at token level"
        )
    elif duration_offsets is not None:
        dur_voc_idx = (
            tokenizer.vocab_types_idx["Duration"] if tokenizer.is_multi_voc else None
        )
        if not tokenizer.is_multi_voc:
            dur_tokens = np.array(tokenizer.token_ids_of_type("Duration"))
        else:
            dur_tokens = np.array(tokenizer.token_ids_of_type("Duration", dur_voc_idx))

        def augment_dur(
            seq_: np.ndarray, offsets_: Tuple[int, int, int]
        ) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
            if not tokenizer.is_multi_voc:
                mask = np.isin(seq_, dur_tokens)
            else:
                mask = np.full_like(seq_, 0, dtype=np.bool_)
                mask[:, dur_voc_idx] = np.isin(seq_[:, dur_voc_idx], dur_tokens)

            aug_ = []
            for offset_ in duration_offsets:
                aug_seq = seq_.copy()
                aug_seq[mask] += offset_
                aug_seq[mask] = np.clip(aug_seq[mask], dur_tokens[0], dur_tokens[-1])
                aug_.append(((offsets_[0], offsets_[1], offset_), aug_seq))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, seq_aug = augmented[i]
                augmented += augment_dur(
                    seq_aug, offsets
                )  # for already augmented midis
        augmented += augment_dur(tokens, (0, 0, 0))  # for original midi

    # Convert all arrays to lists and reapply BPE if necessary
    for i in range(len(augmented)):
        augmented[i] = (
            augmented[i][0],
            tokenizer.apply_bpe(augmented[i][1].tolist())
            if bpe_decoded
            else augmented[i][1].tolist(),
        )

    return augmented
