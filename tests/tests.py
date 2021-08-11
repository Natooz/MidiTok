#!/usr/bin/python3 python

""" Tests file

"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union

from miditok import REMIEncoding, StructuredEncoding, MIDILikeEncoding, CPWordEncoding, MIDIEncoding
from miditoolkit import MidiFile


def one_track_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath] = './Maestro_MIDIs'):
    """
    Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed
    """
    files = list(Path(data_path).glob('**/*.mid'))

    # Creates tokenizers
    cp_enc = CPWordEncoding()
    remi_enc = REMIEncoding()
    struct_enc = StructuredEncoding()
    midilike_enc = MIDILikeEncoding()

    for i, file_path in enumerate(files):
        print(f'Converting MIDI {i} / {len(files)} - {file_path}')

        # Reads the midi
        midi = MidiFile(file_path)
        original_track = deepcopy(midi.instruments[0])

        # Convert the track in tokens
        tokens_cp, _ = cp_enc.midi_to_tokens(midi)
        tokens_remi, _ = remi_enc.midi_to_tokens(midi)
        tokens_struct, _ = struct_enc.midi_to_tokens(midi)
        tokens_midilike, _ = midilike_enc.midi_to_tokens(midi)

        # Convert back tokens into a track object
        track_cp = cp_enc.tokens_to_track(tokens_cp[0], midi.ticks_per_beat)
        track_remi = remi_enc.tokens_to_track(tokens_remi[0], midi.ticks_per_beat)
        track_struct = struct_enc.tokens_to_track(tokens_struct[0], midi.ticks_per_beat)
        track_midilike = midilike_enc.tokens_to_track(tokens_midilike[0], midi.ticks_per_beat)
        track_cp.name = 'encoded with cp word'
        track_remi.name = 'encoded with remi'
        track_struct.name = 'encoded with structured'
        track_midilike.name = 'encoded with midi-like'
        midi.instruments[0].name = 'original quantized'
        original_track.name = 'original not quantized'

        # Updates the MIDI and save it
        midi.instruments += [track_cp, track_remi, track_struct, track_midilike, original_track]
        midi.dump(PurePath('data', 'test_conv', str(i)).with_suffix('.mid'))


def midi_to_tokens_to_midi(tokenizer: MIDIEncoding, midi: MidiFile) -> MidiFile:
    """ Converts a MIDI into tokens, and convert them back to MIDI
    Useful to see if the conversion works well in both ways

    :param tokenizer: the tokenizer
    :param midi: MIDI object to convert
    :return: The converted MIDI object
    """
    tokens, inf = tokenizer.midi_to_tokens(midi)
    new_midi = tokenizer.tokens_to_midi(tokens, inf, time_division=midi.ticks_per_beat)

    return new_midi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIDI Encoding test')
    parser.add_argument('--data', type=str, default='./',
                        help='directory of MIDI files to use for test')
    args = parser.parse_args()

    one_track_midi_to_tokens_to_midi(args.data)
