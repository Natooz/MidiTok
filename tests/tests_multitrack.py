#!/usr/bin/python3 python

""" Multitrack test file
This test method will encode every track of a MIDI file.
These file contains tracks with long empty sections with no notes. Hence encodings
in which time is based on time-shift tokens (MIDI-like, Structured) will probably
not be suited for these files.

Structured and MIDI-Like are then not tested here.
You can still manage to make them work and pass the test be using a vocabulary
with large duration / time-shift values, but this is probably not suited for real
case situations.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

import time
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union

import miditok
from miditoolkit import MidiFile

from tests_utils import midis_equals, tempo_changes_equals

# Special beat res for test, up to 16 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 16): 8}
ADDITIONAL_TOKENS_TEST = {'Chord': True,
                          'Rest': True,
                          'Tempo': True,
                          'rest_range': (16, 16),
                          'nb_tempos': 32,
                          'tempo_range': (40, 250)}


def multitrack_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath] = './Maestro_MIDIs',
                                      saving_midi: bool = True):
    """ Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    """
    encodings = ['REMIEncoding', 'CPWordEncoding', 'OctupleEncoding', 'OctupleMonoEncoding', 'MuMIDIEncoding']
    files = list(Path(data_path).glob('**/*.mid'))

    for i, file_path in enumerate(files):
        print(f'Converting MIDI {i+1} / {len(files)} - {file_path}')

        # Reads the MIDI
        midi = MidiFile(file_path)

        t0 = time.time()
        for encoding in encodings:
            tokenizer = getattr(miditok, encoding)(beat_res=BEAT_RES_TEST,
                                                   additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))

            # MIDI -> Tokens -> MIDI
            new_midi = midi_to_tokens_to_midi(tokenizer, midi)
            midi_to_compare = deepcopy(midi)  # midi is quantized after line above

            # Sort and merge tracks if needed
            # MIDI produced with Octuple contains tracks ordered by program
            if encoding == 'OctupleEncoding' or encoding == 'MuMIDIEncoding':
                miditok.merge_same_program_tracks(midi_to_compare.instruments)  # merge tracks
            if encoding == 'OctupleEncoding':
                sorted_instruments = []  # sort tracks
                for track in midi_to_compare.instruments:
                    for new_track in new_midi.instruments:
                        if new_track.program == track.program:
                            sorted_instruments.append(new_track)
                new_midi.instruments = sorted_instruments

            # Checks notes
            errors = midis_equals(midi_to_compare, new_midi)
            if len(errors) > 0:
                print(f'Failed to encode/decode MIDI with {encoding[:-8]} ({sum(len(t) for t in errors)} errors)')
                # return False

            # Checks tempos
            if tokenizer.additional_tokens['Tempo'] and encoding != 'MuMIDIEncoding':  # MuMIDI doesn't decode tempos
                tempo_errors = tempo_changes_equals(midi_to_compare.tempo_changes, new_midi.tempo_changes)
                if len(tempo_errors) > 0:
                    print(f'Failed to encode/decode TEMPO changes with {encoding[:-8]} ({len(tempo_errors)} errors)')

            if saving_midi:
                new_midi.dump(PurePath('tests', 'test_results', f'{file_path.stem}_{encoding[:-8]}')
                              .with_suffix('.mid'))

        t1 = time.time()
        print(f'Took {t1 - t0} seconds')
    return True


def midi_to_tokens_to_midi(tokenizer: miditok.MIDITokenizer, midi: MidiFile) -> MidiFile:
    """ Converts a MIDI into tokens, and convert them back to MIDI
    Useful to see if the conversion works well in both ways

    :param tokenizer: the tokenizer
    :param midi: MIDI object to convert
    :return: The converted MIDI object
    """
    tokens = tokenizer.midi_to_tokens(midi)
    inf = miditok.get_midi_programs(midi)  # programs of tracks
    new_midi = tokenizer.tokens_to_midi(tokens, inf, time_division=midi.ticks_per_beat)

    return new_midi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIDI Encoding test')
    parser.add_argument('--data', type=str, default='tests/Multitrack_MIDIs',
                        help='directory of MIDI files to use for test')
    args = parser.parse_args()

    multitrack_midi_to_tokens_to_midi(args.data)
