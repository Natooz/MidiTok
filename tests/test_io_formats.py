#!/usr/bin/python3 python

"""One track test file
This test method is to be used with MIDI files of one track (like the maestro dataset).
It is mostly useful to measure the performance of encodings where time is based on
time shifts tokens, as these files usually don't contain tracks with very long pauses,
i.e. long duration / time-shift values probably out of range of the tokenizer's vocabulary.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

from copy import deepcopy
from pathlib import Path

import miditok
from miditoolkit import MidiFile

from .tests_utils import ALL_TOKENIZATIONS, midis_equals, reduce_note_durations


# Very large beat resolution range so that it covers all cases as some tracks
# may have very long pauses when the associated instrument is not playing
BEAT_RES_TEST = {(0, 512): 8}
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
    "rest_range": (
        4,
        512,
    ),  # very high value to cover every possible rest in the test files
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": (16, 2),
}


def encode_decode_and_check(tokenizer: miditok.MIDITokenizer, midi: MidiFile):
    # Process the MIDI
    midi_to_compare = deepcopy(midi)
    for track in midi_to_compare.instruments:
        if track.is_drum:
            track.program = 0  # need to be done before sorting tracks per program
    # Sort and merge tracks if needed
    # MIDI produced with unique_track contains tracks with different orders
    if tokenizer.unique_track:
        miditok.utils.merge_same_program_tracks(midi_to_compare.instruments)
    # reduce the duration of notes to long
    for track in midi_to_compare.instruments:
        reduce_note_durations(
            track.notes,
            max(tu[1] for tu in BEAT_RES_TEST) * midi_to_compare.ticks_per_beat,
        )
        miditok.utils.remove_duplicated_notes(track.notes)
    midi_to_compare.instruments.sort(
        key=lambda x: (x.program, x.is_drum)
    )  # sort tracks

    # Convert the midi to tokens, and keeps the ids (integers)
    tokens = tokenizer(midi_to_compare)
    if tokenizer.unique_track:
        tokens = tokens.ids
    else:
        tokens = [stream.ids for stream in tokens]

    # Convert back token ids to a MIDI object
    kwargs = {"time_division": midi.ticks_per_beat}
    if not tokenizer.unique_track:
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
        tokenizer_config = miditok.TokenizerConfig(**TOKENIZER_PARAMS)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            tokenizer_config=tokenizer_config
        )

        at_least_one_error = (
            encode_decode_and_check(tokenizer, midi) or at_least_one_error
        )

        # If TSD, also test in use_programs / unique_track mode
        if tokenization == "TSD":
            tokenizer_config = miditok.TokenizerConfig(**TOKENIZER_PARAMS)
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
