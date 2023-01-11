"""Data augmentation methods

"""

from typing import List, Tuple, Union
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile

from miditok.midi_tokenizer_base import MIDITokenizer


def data_augmentation_dataset(data_path: Union[Path, str],
                              nb_scales_offset: int,
                              out_path: Union[Path, str] = None,
                              tokenizer: MIDITokenizer = None,
                              pitch_range: range = range(0, 128)):
    r"""Perform data augmentation on a whole dataset, on the pitch dimension.
    Drum tracks are not augmented.

    :param data_path: root path to the folder containing tokenized json files.
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param out_path: output path to save the augmented files. Original (non-augmented) MIDIs will be
            saved to this location. If none is given, they will be saved in the same location an the
            data_path. (default: None)
    :param nb_scales_offset: number of pitch scales to perform data augmentation. Has to be given
            if performing augmentation on tokens (default: None).
    :param pitch_range: minimum and maximum pitch to consider (default: range(0, 128)).
    """
    if out_path is None:
        out_path = data_path
    else:
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
    as_tokens = True
    files_paths = list(Path(data_path).glob('**/*.json'))
    if len(files_paths) == 0:
        files_paths = list(Path(data_path).glob('**/*.mid'))
        as_tokens = False

    nb_augmentations, nb_tracks_augmented = 0, 0
    for file_path in tqdm(files_paths, desc='Performing data augmentation'):

        if as_tokens:
            with open(file_path) as json_file:
                file = json.load(json_file)
                tokens, programs = file['tokens'], file['programs']

            # Perform data augmentation for each track
            augmented_tokens = {i: [] for i in range(-nb_scales_offset, nb_scales_offset + 1)}
            del augmented_tokens[0]
            if tokenizer.unique_track:
                tokens = [tokens]
            for track, (_, is_drum) in zip(tokens, programs):
                if is_drum:
                    for i in range(-nb_scales_offset, nb_scales_offset + 1):
                        if i == 0:
                            continue
                        augmented_tokens[i].append(track)
                aug = data_augmentation_tokens(np.array(track), tokenizer, nb_scales_offset)
                for offset, seq in aug:
                    augmented_tokens[offset].append(seq)

            # Save augmented tracks as MIDI
            for offset, seq in augmented_tokens.items():
                if len(seq) == 0:
                    continue
                saving_path = (file_path.parent if out_path is None else out_path) / f'{file_path.stem}_{offset}.json'
                tokenizer.save_tokens(seq, saving_path, programs)
                nb_augmentations += 1
                nb_tracks_augmented += len(seq)

        else:  # as midi
            try:
                midi = MidiFile(Path(file_path))
            except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            augmented_midis = data_augmentation_midi(midi, pitch_range, nb_scales_offset)
            for offset, aug_midi in augmented_midis:
                if len(aug_midi.instruments) == 0:
                    continue
                saving_path = (file_path.parent if out_path is None else out_path) / f'{file_path.stem}_{offset}.mid'
                aug_midi.dump(saving_path)
                nb_augmentations += 1
                nb_tracks_augmented += len(aug_midi.instruments)
            if out_path is not None:
                midi.dump(out_path / f'{file_path.stem}.mid')

    # Saves data augmentation report, json encoded with txt extension to not mess with others json files
    with open(data_path / 'data_augmentation.txt', 'w') as outfile:
        json.dump({'nb_tracks_augmented': nb_tracks_augmented,
                   'nb_files_before': len(files_paths),
                   'nb_files_after': len(files_paths) + nb_augmentations}, outfile)


def data_augmentation_midi(midi: MidiFile, pitch_range: range, nb_scales_offset: int) -> List[Tuple[int, MidiFile]]:
    # Get the maximum and lowest pitch in original track
    all_pitches = []
    for track in midi.instruments:
        if not track.is_drum:
            all_pitches += [note.pitch for note in track.notes]
    max_pitch, min_pitch = max(all_pitches), min(all_pitches)
    offset_up = min(nb_scales_offset, (pitch_range.stop - 1 - int(max_pitch)) // 12)
    offset_down = min(nb_scales_offset, (int(min_pitch) - pitch_range.start) // 12)

    # Perform augmentation on pitch
    augmented = []
    for i in range(offset_up):  # UP
        midi_augmented = deepcopy(midi)
        for track in midi_augmented.instruments:
            if not track.is_drum:
                for note in track.notes:
                    note.pitch += (i + 1) * 12
        augmented.append((i + 1, midi_augmented))
    for i in range(offset_down):  # DOWN
        midi_augmented = deepcopy(midi)
        for track in midi_augmented.instruments:
            if not track.is_drum:
                for note in track.notes:
                    note.pitch -= (i + 1) * 12
        augmented.append((- (i + 1), midi_augmented))

    return augmented


def data_augmentation_tokens(tokens: Union[np.ndarray, List[int]], tokenizer: MIDITokenizer, nb_scales_offset: int) \
        -> List[Tuple[int, List[int]]]:
    r"""Perform data augmentation on a sequence of tokens, on the pitch dimension.
    NOTE: token sequences with BPE will be decoded during the augmentation, this might take some time.
    NOTE 2: the tokenizer must have a vocabulary in which the pitch values increase with the token index,
    e.g. Pitch_48: token 64, Pitch_49: token65 ...
    The same goes for durations.
    MIDILike is not compatible with data augmentation on durations.
    MuMIDI is not compatible at all.

    :param tokens: tokens to perform data augmentation on.
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_scales_offset: number of pitch scales to perform data augmentation.
    :return: the several data augmentations that have been performed
    """
    # Decode BPE
    bpe_decoded = False
    if tokenizer.has_bpe:
        tokens = tokenizer.decompose_bpe(tokens.tolist() if isinstance(tokens, np.ndarray) else tokens)
        bpe_decoded = True

    # Converts to np array if necessary
    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    # Get the maximum and lowest pitch in original track
    pitch_voc_idx = tokenizer.vocab_types_idx['Pitch'] if tokenizer.is_multi_voc else None
    tokens_pitch, tokens_pitch_idx, note_off_idx = [], [], []
    note_off_tokens = []
    if not tokenizer.is_multi_voc:
        pitch_tokens = np.array(tokenizer.vocab.tokens_of_type('Pitch') + tokenizer.vocab.tokens_of_type('NoteOn'))
        note_off_tokens = np.array(tokenizer.vocab.tokens_of_type('NoteOff'))
        for i, tok in enumerate(tokens):
            if tok in pitch_tokens:
                tokens_pitch.append(tok)
                tokens_pitch_idx.append(i)
            if len(note_off_tokens) > 0:  # for MIDILike
                if tok in note_off_tokens:
                    note_off_idx.append(i)
        max_pitch = tokenizer[int(np.max(tokens_pitch))].split('_')[1]
        min_pitch = tokenizer[int(np.min(tokens_pitch))].split('_')[1]
    else:
        pitch_tokens = np.array(tokenizer.vocab[pitch_voc_idx].tokens_of_type('Pitch'))
        special_tokens = ['Pitch_Ignore']
        special_tokens = [tokenizer.vocab[pitch_voc_idx][t] for t in special_tokens
                          if t in list(tokenizer.vocab[pitch_voc_idx].event_to_token.keys())]
        for i, tok in enumerate(tokens):
            if tok[pitch_voc_idx] in pitch_tokens and tok[pitch_voc_idx] not in special_tokens:
                tokens_pitch.append(tok[pitch_voc_idx])
                tokens_pitch_idx.append(i)
        max_pitch = tokenizer.vocab[pitch_voc_idx][int(max(tokens_pitch))].split('_')[1]
        min_pitch = tokenizer.vocab[pitch_voc_idx][int(min(tokens_pitch))].split('_')[1]
    offset_up = min(nb_scales_offset, (tokenizer.pitch_range.stop - 1 - int(max_pitch)) // 12)
    offset_down = min(nb_scales_offset, (int(min_pitch) - tokenizer.pitch_range.start) // 12)

    # Perform augmentation on pitch
    augmented = []
    for i in range(offset_up):
        seq = tokens.copy()
        if not tokenizer.is_multi_voc:
            seq[tokens_pitch_idx] += (i + 1) * 12  # shifts pitches scale up
            if len(note_off_tokens) > 0:  # for MIDIlike
                seq[note_off_idx] += (i + 1) * 12
        else:
            seq[tokens_pitch_idx, np.full((len(tokens_pitch_idx),), pitch_voc_idx)] += (i + 1) * 12  # shifts scale up
        augmented.append(((i + 1), seq.tolist()))
    for i in range(offset_down):
        seq = tokens.copy()
        if not tokenizer.is_multi_voc:
            seq[tokens_pitch_idx] -= (i + 1) * 12  # shifts pitches scale down
            if len(note_off_tokens) > 0:  # for MIDIlike
                seq[note_off_idx] -= (i + 1) * 12
        else:
            seq[tokens_pitch_idx, np.full((len(tokens_pitch_idx),), pitch_voc_idx)] -= (i + 1) * 12  # shifts scale down
        augmented.append((- (i + 1), seq.tolist()))

    '''if type(tokenizer).__name__ == 'MIDILike':
        warn(f'{type(tokenizer).__name__} is not compatible with data augmentation on duration')'''

    # Reapply BPE on all sequences
    if bpe_decoded:
        for i in range(len(augmented)):
            augmented[i] = (augmented[i][0], tokenizer.apply_bpe(augmented[i][1].tolist()))

    return augmented
