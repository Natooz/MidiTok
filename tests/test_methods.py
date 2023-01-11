#!/usr/bin/python3 python

"""Test methods

"""

from pathlib import Path

import miditok
from miditoolkit import MidiFile
from torch import Tensor as ptTensor, IntTensor as ptIntTensor, FloatTensor as ptFloatTensor
from tensorflow import Tensor as tfTensor, convert_to_tensor
from tqdm import tqdm


def test_convert_tensors():
    original = [[2, 6, 87, 89, 25, 15]]
    types = [ptTensor, ptIntTensor, ptFloatTensor, tfTensor]

    def nothing(tokens):
        return tokens

    tokenizer = miditok.TSD()
    for type_ in types:
        if type_ == tfTensor:
            tensor = convert_to_tensor(original)
        else:
            tensor = type_(original)
        tokenizer(tensor)  # to make sure it passes
        as_list = miditok.midi_tokenizer_base.convert_tokens_tensors_to_list(nothing)(tensor)
        assert as_list == original


def time_data_augmentation_tokens_vs_mid():
    from time import time
    tokenizer = miditok.TSD()
    data_path = Path('./tests/Maestro_MIDIs')
    files = list(data_path.glob('**/*.mid'))

    # Testing opening midi -> augment midis -> tokenize midis
    t0 = time()
    for file_path in files:
        # Reads the MIDI
        try:
            midi = MidiFile(Path(file_path))
        except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            continue

        midis = miditok.data_augmentation.data_augmentation_midi(midi, tokenizer.pitch_range, 2)
        for _, aug_mid in midis:
            _ = tokenizer(aug_mid)
    print(f'Opening midi -> augment midis -> tokenize midis: took {(tt := time() - t0):.2f} sec '
          f'({tt / len(files):.2f} sec/file)')

    # Testing opening midi -> tokenize midi -> augment tokens
    t0 = time()
    for file_path in files:
        # Reads the MIDI
        try:
            midi = MidiFile(Path(file_path))
        except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            continue

        tokens = tokenizer(midi)
        for track_tokens in tokens:
            _ = miditok.data_augmentation.data_augmentation_tokens(track_tokens, tokenizer, 2)
    print(f'Opening midi -> tokenize midi -> augment tokens: took {(tt := time() - t0):.2f} sec '
          f'({tt / len(files):.2f} sec/file)')


def test_data_augmentation():
    data_path = Path('./tests/Multitrack_MIDIs')
    tokenizations = ['TSD', 'MIDILike', 'REMI', 'Structured', 'CPWord', 'Octuple', 'OctupleMono']
    files = list(data_path.glob('**/*.mid'))

    for tokenization in tqdm(tokenizations, desc='Testing data augmentation'):
        tokenizer = getattr(miditok, tokenization)()

        miditok.data_augmentation.data_augmentation_dataset(data_path, 2, Path('./tests/Multitrack_MIDIs_aug'),
                                                            tokenizer, tokenizer.pitch_range)  # as midi
        tokenizer.tokenize_midi_dataset(files, Path('./tests/Multitrack_tokens'))
        miditok.data_augmentation.data_augmentation_dataset(Path('./tests/Multitrack_tokens'), 2,
                                                            Path('./tests/Multitrack_tokens_aug'), tokenizer)

        for file_path in files:
            # Reads the MIDI
            try:
                midi = MidiFile(Path(file_path))
            except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # MIDI -> Tokens
            tokens = tokenizer(midi)

            # Perform data augmentation
            if not tokenizer.unique_track:  # track per track
                for track_seq in tokens:
                    augmented = miditok.data_augmentation.data_augmentation_tokens(track_seq, tokenizer, 2)
                    non_shifted_track = tokenizer([track_seq]).instruments[0]
                    shifted_midi = tokenizer([seq for _, seq in augmented])

                    # Checks notes
                    for (offset, _), shifted_track in zip(augmented, shifted_midi.instruments):
                        for note_o, note_s in zip(non_shifted_track.notes, shifted_track.notes):
                            assert note_s.pitch == note_o.pitch + offset * 12
            else:  # all tracks in parallel (multitrack)
                augmented = miditok.data_augmentation.data_augmentation_tokens(tokens, tokenizer, 2)
                non_shifted_midi = tokenizer(tokens)
                shifted_midis = [tokenizer(seq) for _, seq in augmented]

                # Checks notes
                for (offset, _), shifted_midi in zip(augmented, shifted_midis):
                    for non_shifted_track, shifted_track in zip(non_shifted_midi.instruments, shifted_midi.instruments):
                        for note_o, note_s in zip(non_shifted_track.notes, shifted_track.notes):
                            assert note_s.pitch == note_o.pitch + offset * 12


if __name__ == "__main__":
    test_convert_tensors()
    test_data_augmentation()
