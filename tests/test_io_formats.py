#!/usr/bin/python3 python

"""
Testing the possible I/O formats of the tokenizers.
"""

from copy import deepcopy
from pathlib import Path

import miditok
from miditoolkit import MidiFile

from .tests_utils import ALL_TOKENIZATIONS, midis_equals


BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "beat_res_rest": {(0, 16): 4},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": {4: [4]},
}


def encode_decode_and_check(tokenizer: miditok.MIDITokenizer, midi: MidiFile):
    # Process the MIDI
    midi_to_compare = deepcopy(midi)
    for track in midi_to_compare.instruments:
        if track.is_drum:
            track.program = 0  # need to be done before sorting tracks per program
    # MIDI produced with one_token_stream contains tracks with different orders
    midi_to_compare.instruments.sort(
        key=lambda x: (x.program, x.is_drum)
    )  # sort tracks

    # Convert the midi to tokens, and keeps the ids (integers)
    tokens = tokenizer(midi_to_compare)
    if tokenizer.one_token_stream:
        tokens = tokens.ids
    else:
        tokens = [stream.ids for stream in tokens]

    # Convert back token ids to a MIDI object
    kwargs = {"time_division": midi.ticks_per_beat}
    if not tokenizer.one_token_stream:
        kwargs["programs"] = miditok.utils.get_midi_programs(midi_to_compare)
    try:
        decoded_midi = tokenizer(tokens, **kwargs)
    except Exception as e:
        print(f"Error when decoding token ids with {tokenizer.__class__.__name__}: {e}")
        return True

    # Checks its good
    """for track1, track2 in zip(midi_to_compare.instruments, decoded_midi.instruments):
        if len(track1.notes) != len(track2.notes):
            at_least_one_error = True
            break"""
    decoded_midi.instruments.sort(key=lambda x: (x.program, x.is_drum))
    errors = midis_equals(midi_to_compare, decoded_midi)
    if len(errors) > 0:
        print(
            f"Failed to encode/decode NOTES with {tokenizer.__class__.__name__} ({len(errors)} errors)"
        )
        return True

    return False


def test_io_formats():
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed
    """
    at_least_one_error = False

    file_path = Path("tests", "Multitrack_MIDIs", "Funkytown.mid")
    midi = MidiFile(file_path)

    for tokenization in ALL_TOKENIZATIONS:
        params = deepcopy(TOKENIZER_PARAMS)
        if tokenization == "Structured":
            params["beat_res"] = {(0, 512): 8}
        tokenizer_config = miditok.TokenizerConfig(**params)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            tokenizer_config=tokenizer_config
        )

        at_least_one_error = (
            encode_decode_and_check(tokenizer, midi) or at_least_one_error
        )

        # If TSD, also test in use_programs / one_token_stream mode
        if tokenization in ["TSD", "REMI", "MIDILike", "Structured", "CPWord"]:
            tokenizer_config = miditok.TokenizerConfig(**params)
            tokenizer_config.use_programs = True
            tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
                tokenizer_config=tokenizer_config
            )
            at_least_one_error = (
                encode_decode_and_check(tokenizer, midi) or at_least_one_error
            )

    assert not at_least_one_error


if __name__ == "__main__":
    test_io_formats()
