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

from miditok import REMIEncoding, StructuredEncoding, MIDILikeEncoding, CPWordEncoding, MuMIDIEncoding, OctupleEncoding
from miditoolkit import MidiFile

# Special beat res for test, up to 32 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 32): 8}
ADDITIONAL_TOKENS_TEST = {'Chord': False,
                          'Empty': False,
                          'Tempo': False,
                          'Ignore': True}  # for CP words only


def one_track_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath] = './Maestro_MIDIs',
                                     saving_midi: bool = True):
    """ Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    """
    files = list(Path(data_path).glob('**/*.mid'))

    # Creates tokenizers
    cp_enc = CPWordEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))
    remi_enc = REMIEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))
    struct_enc = StructuredEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))
    midilike_enc = MIDILikeEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))
    mumidi_enc = MuMIDIEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))
    oct_enc = OctupleEncoding(beat_res=BEAT_RES_TEST, additional_tokens=deepcopy(ADDITIONAL_TOKENS_TEST))

    for i, file_path in enumerate(files):
        t0 = time.time()
        print(f'Converting MIDI {i} / {len(files)} - {file_path}')

        # Reads the midi
        midi = MidiFile(file_path)
        original_track = deepcopy(midi.instruments[0])

        # Convert the track in tokens
        tokens_cp, _ = cp_enc.midi_to_tokens(midi)
        tokens_remi, _ = remi_enc.midi_to_tokens(midi)
        tokens_struct, _ = struct_enc.midi_to_tokens(midi)
        tokens_midilike, _ = midilike_enc.midi_to_tokens(midi)
        tokens_mumidi = mumidi_enc.midi_to_tokens(midi)
        tokens_oct = oct_enc.midi_to_tokens(midi)

        # Convert back tokens into a track object
        track_cp = cp_enc.tokens_to_track(tokens_cp[0], midi.ticks_per_beat)
        track_remi = remi_enc.tokens_to_track(tokens_remi[0], midi.ticks_per_beat)
        track_struct = struct_enc.tokens_to_track(tokens_struct[0], midi.ticks_per_beat)
        track_midilike = midilike_enc.tokens_to_track(tokens_midilike[0], midi.ticks_per_beat)
        track_mumidi = mumidi_enc.tokens_to_midi(tokens_mumidi, time_division=midi.ticks_per_beat).instruments[0]
        track_oct = oct_enc.tokens_to_midi(tokens_oct, time_division=midi.ticks_per_beat).instruments[0]

        t1 = time.time()
        print(f'Took {t1 - t0} seconds')

        if saving_midi:
            track_cp.name = 'encoded with cp word'
            track_remi.name = 'encoded with remi'
            track_struct.name = 'encoded with structured'
            track_midilike.name = 'encoded with midi-like'
            track_mumidi.name = 'encoded with mumidi'
            track_oct.name = 'encoded with octuple'
            midi.instruments[0].name = 'original quantized'
            original_track.name = 'original not quantized'

            # Updates the MIDI and save it
            midi.instruments += [track_cp, track_remi, track_struct, track_midilike, track_mumidi, track_oct,
                                 original_track]
            midi.dump(PurePath('tests', 'test_results', file_path.name))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIDI Encoding test')
    parser.add_argument('--data', type=str, default='tests/Maestro_MIDIs',
                        help='directory of MIDI files to use for test')
    args = parser.parse_args()

    one_track_midi_to_tokens_to_midi(args.data)
