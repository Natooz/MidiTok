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

from pathlib import Path, PurePath
from typing import Union

from miditok import REMIEncoding, CPWordEncoding, MIDIEncoding
from miditoolkit import MidiFile


def multitrack_midi_to_tokens_to_midi(data_path: Union[str, Path, PurePath] = './Maestro_MIDIs',
                                      saving_midi: bool = True):
    """ Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    """
    files = list(Path(data_path).glob('**/*.mid'))

    # Creates tokenizers
    cp_enc = CPWordEncoding()
    remi_enc = REMIEncoding()

    for i, file_path in enumerate(files):
        print(f'Converting MIDI {i} / {len(files)} - {file_path}')

        # Reads the MIDI
        midi = MidiFile(file_path)

        # Convert to tokens and back to MIDI
        midi_cp = midi_to_tokens_to_midi(cp_enc, midi)
        midi_remi = midi_to_tokens_to_midi(remi_enc, midi)

        if saving_midi:
            # Build the final MIDI file
            final_midi = MidiFile()
            for j in range(len(midi.instruments)):
                midi_cp.instruments[j].name = 'encoded with cp word'
                midi_remi.instruments[j].name = 'encoded with remi'

                final_midi.instruments.append(midi.instruments[j])
                final_midi.instruments.append(midi_cp.instruments[j])
                final_midi.instruments.append(midi_remi.instruments[j])

            # Updates the MIDI and save it
            final_midi.dump(PurePath('tests', 'test_results', file_path.name))


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
    parser.add_argument('--data', type=str, default='tests/Multitrack_MIDIs',
                        help='directory of MIDI files to use for test')
    args = parser.parse_args()

    multitrack_midi_to_tokens_to_midi(args.data)
