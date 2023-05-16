#!/usr/bin/python3 python

"""Test methods

"""

from pathlib import Path
import json

import numpy as np
from miditoolkit import MidiFile
from torch import (
    Tensor as ptTensor,
    IntTensor as ptIntTensor,
    FloatTensor as ptFloatTensor,
)
from tensorflow import Tensor as tfTensor, convert_to_tensor

import miditok


def test_convert_tensors():
    original = [[2, 6, 87, 89, 25, 15]]
    types = [ptTensor, ptIntTensor, ptFloatTensor, tfTensor]

    tokenizer = miditok.TSD()
    for type_ in types:
        if type_ == tfTensor:
            tensor = convert_to_tensor(original)
        else:
            tensor = type_(original)
        tokenizer(tensor)  # to make sure it passes as decorator
        as_list = miditok.midi_tokenizer.convert_ids_tensors_to_list(tensor)
        assert as_list == original


"""def time_data_augmentation_tokens_vs_mid():
    from time import time
    tokenizers = [miditok.TSD(), miditok.REMI()]
    data_paths = [Path('./tests/Maestro_MIDIs'), Path('./tests/Multitrack_MIDIs')]

    for data_path in data_paths:
        for tokenizer in tokenizers:
            print(f'\n{data_path.stem} - {type(tokenizer).__name__}')
            files = list(data_path.glob('**/*.mid'))

            # Testing opening midi -> augment midis -> tokenize midis
            t0 = time()
            for file_path in files:
                # Reads the MIDI
                try:
                    midi = MidiFile(Path(file_path))
                except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                    continue

                offsets = miditok.data_augmentation.get_offsets(tokenizer, 2, 2, 2, midi=midi)
                midis = miditok.data_augmentation.data_augmentation_midi(midi, tokenizer, *offsets)
                for _, aug_mid in midis:
                    _ = tokenizer(aug_mid)
            tt = time() - t0
            print(f'Opening midi -> augment midis -> tokenize midis: took {tt:.2f} sec '
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
                    offsets = miditok.data_augmentation.get_offsets(tokenizer, 2, 2, 2, tokens=tokens)
                    _ = miditok.data_augmentation.data_augmentation_tokens(track_tokens, tokenizer, *offsets)
            tt = time() - t0
            print(f'Opening midi -> tokenize midi -> augment tokens: took {tt:.2f} sec '
                  f'({tt / len(files):.2f} sec/file)')"""


def test_data_augmentation():
    data_path = Path("./tests/Multitrack_MIDIs")
    tokenizations = [
        "TSD",
        "MIDILike",
        "REMI",
        "REMIPlus",
        "Structured",
        "CPWord",
        "Octuple",
        "OctupleMono",
    ]
    original_midi_paths = list(data_path.glob("**/*.mid"))

    for tokenization in tokenizations:
        print(f"TESTING WITH {tokenization}")
        tokenizer = getattr(miditok, tokenization)()
        midi_aug_path = Path("tests", "Multitrack_MIDIs_aug", tokenization)
        tokens_path = Path("tests", "Multitrack_tokens", tokenization)
        tokens_aug_path = Path("tests", "Multitrack_tokens_aug", tokenization)

        print("PERFORMING DATA AUGMENTATION ON MIDIS")
        miditok.data_augmentation.data_augmentation_dataset(
            data_path,
            tokenizer,
            2,
            1,
            1,
            out_path=midi_aug_path,
            copy_original_in_new_location=False,
        )
        aug_midi_paths = list(midi_aug_path.glob("**/*.mid"))
        for aug_midi_path in aug_midi_paths:
            # Determine offsets of file
            parts = aug_midi_path.stem.split("§")
            original_stem, offsets_str = parts[0], parts[1].split("_")
            offsets = [0, 0, 0]
            for offset_str in offsets_str:
                for pos, letter in enumerate(["p", "v", "d"]):
                    if offset_str[0] == letter:
                        offsets[pos] = int(offset_str[1:])

            # Loads MIDIs to compare
            try:
                aug_midi = MidiFile(aug_midi_path)
                original_midi = MidiFile(data_path / f"{original_stem}.mid")
            except (
                Exception
            ):  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # Compare them
            for original_track, aug_track in zip(
                original_midi.instruments, aug_midi.instruments
            ):
                if original_track.is_drum:
                    continue
                original_track.notes.sort(
                    key=lambda x: (x.start, x.pitch, x.end, x.velocity)
                )  # sort notes
                aug_track.notes.sort(
                    key=lambda x: (x.start, x.pitch, x.end, x.velocity)
                )  # sort notes
                for note_o, note_s in zip(original_track.notes, aug_track.notes):
                    assert note_s.pitch == note_o.pitch + offsets[0]
                    assert note_s.velocity in [
                        tokenizer.velocities[0],
                        tokenizer.velocities[-1],
                        note_o.velocity + offsets[1],
                    ]

        print("PERFORMING DATA AUGMENTATION ON TOKENS")
        tokenizer.tokenize_midi_dataset(original_midi_paths, tokens_path)
        miditok.data_augmentation.data_augmentation_dataset(
            tokens_path,
            tokenizer,
            2,
            1,
            1,
            out_path=tokens_aug_path,
            copy_original_in_new_location=False,
        )

        # Getting tokens idx from tokenizer for assertions
        aug_tokens_paths = list(tokens_aug_path.glob("**/*.json"))
        pitch_voc_idx, vel_voc_idx, dur_voc_idx = None, None, None
        note_off_tokens = []
        if tokenizer.is_multi_voc:
            pitch_voc_idx = tokenizer.vocab_types_idx["Pitch"]
            vel_voc_idx = tokenizer.vocab_types_idx["Velocity"]
            dur_voc_idx = tokenizer.vocab_types_idx["Duration"]
            pitch_tokens = np.array(tokenizer.token_ids_of_type("Pitch", pitch_voc_idx))
            vel_tokens = np.array(tokenizer.token_ids_of_type("Velocity", vel_voc_idx))
            dur_tokens = np.array(tokenizer.token_ids_of_type("Duration", dur_voc_idx))
        else:
            pitch_tokens = np.array(
                tokenizer.token_ids_of_type("Pitch")
                + tokenizer.token_ids_of_type("NoteOn")
            )
            vel_tokens = np.array(tokenizer.token_ids_of_type("Velocity"))
            dur_tokens = np.array(tokenizer.token_ids_of_type("Duration"))
            note_off_tokens = np.array(
                tokenizer.token_ids_of_type("NoteOff")
            )  # for MidiLike
        tok_vel_min, tok_vel_max = vel_tokens[0], vel_tokens[-1]
        tok_dur_min, tok_dur_max = None, None
        if tokenization != "MIDILike":
            tok_dur_min, tok_dur_max = dur_tokens[0], dur_tokens[-1]

        for aug_token_path in aug_tokens_paths:
            # Determine offsets of file
            parts = aug_token_path.stem.split("§")
            original_stem, offsets_str = parts[0], parts[1].split("_")
            offsets = [0, 0, 0]
            for offset_str in offsets_str:
                for pos, letter in enumerate(["p", "v", "d"]):
                    if offset_str[0] == letter:
                        offsets[pos] = int(offset_str[1:])

            # Loads tokens to compare
            with open(aug_token_path) as json_file:
                file = json.load(json_file)
                aug_tokens, aug_programs = file["ids"], file["programs"]
            with open(tokens_path / f"{original_stem}.json") as json_file:
                file = json.load(json_file)
                original_tokens, original_programs = file["ids"], file["programs"]

            # Compare them
            if tokenizer.unique_track:
                original_tokens, aug_tokens = [original_tokens], [aug_tokens]
            for original_track, aug_track, (_, is_drum) in zip(
                original_tokens, aug_tokens, original_programs
            ):
                if is_drum:
                    continue
                for original_token, aug_token in zip(original_track, aug_track):
                    if not tokenizer.is_multi_voc:
                        if original_token in pitch_tokens:
                            assert aug_token == original_token + offsets[0]
                        elif original_token in vel_tokens:
                            assert aug_token in [
                                original_token + offsets[1],
                                tok_vel_min,
                                tok_vel_max,
                            ]
                        elif (
                            original_token in dur_tokens and tokenization != "MIDILike"
                        ):
                            assert aug_token in [
                                original_token + offsets[2],
                                tok_dur_min,
                                tok_dur_max,
                            ]
                        elif original_token in note_off_tokens:
                            assert aug_token == original_token + offsets[0]
                    else:
                        if original_token[pitch_voc_idx] in pitch_tokens:
                            assert (
                                aug_token[pitch_voc_idx]
                                == original_token[pitch_voc_idx] + offsets[0]
                            )
                        elif original_token[vel_voc_idx] in vel_tokens:
                            assert aug_token[vel_voc_idx] in [
                                original_token[vel_voc_idx] + offsets[1],
                                tok_vel_min,
                                tok_vel_max,
                            ]
                        elif (
                            original_token[dur_voc_idx] in dur_tokens
                            and tokenization != "MIDILike"
                        ):
                            assert aug_token[dur_voc_idx] in [
                                original_token[dur_voc_idx] + offsets[2],
                                tok_dur_min,
                                tok_dur_max,
                            ]


if __name__ == "__main__":
    test_convert_tensors()
    test_data_augmentation()
