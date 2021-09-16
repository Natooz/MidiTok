#!/usr/bin/python3 python

""" One track test file
This test method is to be used with MIDI files of one track (like the maestro dataset).
It is mostly useful to measure the performance of encodings where time is based on
time shifts tokens, as these files usually don't contain tracks with very long pauses,
i.e. long duration / time-shift values probably out of range of the tokenizer's vocabulary.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

import time
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union

import miditok
from miditoolkit import MidiFile, Marker

from tests_utils import track_equals, tempo_changes_equals

# Special beat res for test, up to 64 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 64): 8}
ADDITIONAL_TOKENS_TEST = {'Chord': True,  # set to false to speed up tests as it takes some time on maestro MIDIs
                          'Rest': True,
                          'Tempo': True,
                          'rest_range': (16, 16),
                          'nb_tempos': 32,
                          'tempo_range': (40, 250)}


def one_track_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath] = './Maestro_MIDIs',
                                     saving_midi: bool = True) -> bool:
    """ Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    :param data_path: root path to the data to test
    :param saving_midi: whether to save the results in a MIDI file
    """
    encodings = ['MIDILikeEncoding', 'StructuredEncoding', 'REMIEncoding', 'CPWordEncoding', 'OctupleEncoding',
                 'OctupleMonoEncoding', 'MuMIDIEncoding']
    files = list(Path(data_path).glob('**/*.mid'))

    for i, file_path in enumerate(files):
        print(f'Converting MIDI {i + 1} / {len(files)} - {file_path}')

        # Reads the midi
        midi = MidiFile(file_path)
        tracks = [deepcopy(midi.instruments[0])]

        t0 = time.time()
        for encoding in encodings:
            if encoding == 'StructuredEncoding':
                tokenizer = getattr(miditok, encoding)(beat_res=BEAT_RES_TEST)
            else:
                tokenizer = getattr(miditok, encoding)(beat_res=BEAT_RES_TEST,
                                                       additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))

            # Convert the track in tokens
            tokens = tokenizer.midi_to_tokens(midi)

            # Convert back tokens into a track object
            tempo_changes = None
            if encoding == 'OctupleEncoding' or encoding == 'MuMIDIEncoding':
                new_midi = tokenizer.tokens_to_midi(tokens, time_division=midi.ticks_per_beat)
                track = new_midi.instruments[0]
                if encoding == 'OctupleEncoding':
                    tempo_changes = new_midi.tempo_changes
            else:
                track, tempo_changes = tokenizer.tokens_to_track(tokens[0], midi.ticks_per_beat)

            # Checks its good
            errors = track_equals(midi.instruments[0], track)
            if len(errors) > 0:
                if errors[0][0] != 'len':
                    for err, note in errors:
                        midi.markers.append(Marker(f'ERR {encoding[:-8]} with note {err} (pitch {note.pitch})',
                                                   note.start))
                print(f'Failed to encode/decode MIDI with {encoding[:-8]} ({len(errors)} errors)')
                # return False
            track.name = f'encoded with {encoding[:-8]}'
            tracks.append(track)

            # Checks tempos
            if tempo_changes is not None and tokenizer.additional_tokens['Tempo']:
                tempo_errors = tempo_changes_equals(midi.tempo_changes, tempo_changes)
                if len(tempo_errors) > 0:
                    print(f'Failed to encode/decode TEMPO changes with {encoding[:-8]} ({len(tempo_errors)} errors)')

        t1 = time.time()
        print(f'Took {t1 - t0} seconds')

        if saving_midi:
            midi.instruments[0].name = 'original quantized'
            tracks[0].name = 'original not quantized'

            # Updates the MIDI and save it
            midi.instruments += tracks
            midi.dump(PurePath('tests', 'test_results', file_path.name))
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIDI Encoding test')
    parser.add_argument('--data', type=str, default='tests/Maestro_MIDIs',
                        help='directory of MIDI files to use for test')
    args = parser.parse_args()
    one_track_midi_to_tokens_to_midi(args.data)
