#!/usr/bin/python3 python

""" One track test file
This test method is to be used with MIDI files of one track (like the maestro dataset).
It is mostly useful to measure the performance of encodings where time is based on
time shifts tokens, as these files usually don't contain tracks with very long pauses,
i.e. long duration / time-shift values probably out of range of the tokenizer's vocabulary.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
import time
import json

import miditok
from miditoolkit import MidiFile
from tqdm import tqdm

# Special beat res for test, up to 64 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 4): 8, (4, 12): 4}
ADDITIONAL_TOKENS_TEST = {'Chord': False,  # set to false to speed up tests as it takes some time on maestro MIDIs
                          'Rest': True,
                          'Tempo': True,
                          'TimeSignature': True,
                          'Program': False,
                          'rest_range': (4, 16),
                          'nb_tempos': 32,
                          'tempo_range': (40, 250),
                          'time_signature_range': (16, 2)}


def one_track_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath]):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    :param data_path: root path to the data to test
    """
    encodings = ['Structured', 'REMI', 'MIDILike']
    tokenizers = []
    data_path = Path(data_path)
    files = list(data_path.glob('**/*.mid'))
    for encoding in encodings:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizers.append(miditok.bpe(getattr(miditok, encoding), beat_res=BEAT_RES_TEST, additional_tokens=add_tokens))
        tokenizers[-1].tokenize_midi_dataset(files, data_path / encoding)
        tokenizers[-1].bpe(data_path / encoding, len(tokenizers[-1].vocab) + 120, out_dir=data_path / f'{encoding}_bpe',
                           files_lim=None, save_converted_samples=True)

    # Reload (test) tokenizer from the saved config file
    tokenizers = []
    for i, encoding in enumerate(encodings):
        tokenizers.append(miditok.bpe(getattr(miditok, encoding), params=data_path / f'{encoding}_bpe' / 'config.txt'))

    t0 = time.time()
    for i, file_path in enumerate(tqdm(files, desc='Testing')):
        # Reads the midi
        midi = MidiFile(file_path)

        for encoding, tokenizer in zip(encodings, tokenizers):
            # Convert the track in tokens
            tokens = tokenizer.midi_to_tokens(deepcopy(midi))[0]  # with BPE
            with open(data_path / f'{encoding}' / f'{file_path.stem}.json') as json_file:
                tokens_no_bpe = json.load(json_file)['tokens'][0]  # no BPE
            tokens_no_bpe2 = tokenizer.decompose_bpe(deepcopy(tokens))  # BPE decomposed
            with open(data_path / f'{encoding}_bpe' / f'{file_path.stem}.json') as json_file:
                saved_tokens = json.load(json_file)['tokens'][0]  # with BPE, saved after creating vocab
            saved_tokens_decomposed = tokenizer.decompose_bpe(deepcopy(saved_tokens))
            no_error_bpe = tokens == saved_tokens
            no_error = tokens_no_bpe == tokens_no_bpe2 == saved_tokens_decomposed
            if not no_error or not no_error_bpe:
                print(f'error for {encoding} and {file_path.name}')

    print(f'Took {time.time() - t0} seconds')


if __name__ == "__main__":
    one_track_midi_to_tokens_to_midi('tests/Maestro_MIDIs')
