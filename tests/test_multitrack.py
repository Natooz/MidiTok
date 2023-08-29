#!/usr/bin/python3 python

"""Multitrack test file
"""

from copy import deepcopy
from pathlib import Path
from typing import Union
from time import time

import miditok
from miditoolkit import MidiFile, Marker, Pedal
from tqdm import tqdm

from .tests_utils import (
    ALL_TOKENIZATIONS,
    midis_equals,
    tempo_changes_equals,
    pedal_equals,
    pitch_bend_equals,
    adapt_tempo_changes_times,
    time_signature_changes_equals,
    remove_equal_successive_tempos,
)

BEAT_RES_TEST = {(0, 16): 8}
TOKENIZER_PARAMS = {
    "beat_res": BEAT_RES_TEST,
    "use_chords": True,
    "use_rests": True,  # tempo decode fails when False for MIDILike because beat_res range is too short
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "use_programs": True,
    "chord_maps": miditok.constants.CHORD_MAPS,
    "chord_tokens_with_root_note": True,  # Tokens will look as "Chord_C:maj"
    "chord_unknown": (3, 6),
    "beat_res_rest": {(0, 2): 4, (2, 12): 2},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "log_tempos": False,
    "time_signature_range": {4: [3, 4]},
    "sustain_pedal_duration": False,
}


def test_multitrack_midi_to_tokens_to_midi(
    data_path: Union[str, Path] = "./tests/Multitrack_MIDIs",
    saving_erroneous_midis: bool = False,
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    """
    files = list(Path(data_path).glob("**/*.mid"))
    has_errors = False
    t0 = time()

    for i, file_path in enumerate(tqdm(files, desc="Testing multitrack")):
        # Reads the MIDI
        midi = MidiFile(Path(file_path))
        if midi.ticks_per_beat % max(BEAT_RES_TEST.values()) != 0:
            continue
        has_errors = False
        # add pedal messages
        for ti in range(max(3, len(midi.instruments))):
            midi.instruments[ti].pedals = [
                Pedal(start, start + 200) for start in [100, 600, 1800, 2200]
            ]

        for tokenization in ALL_TOKENIZATIONS:
            tokenizer_config = miditok.TokenizerConfig(**TOKENIZER_PARAMS)
            tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
                tokenizer_config=tokenizer_config
            )

            # Process the MIDI
            # midi notes / tempos / time signature quantized with the line above
            midi_to_compare = deepcopy(midi)
            for track in midi_to_compare.instruments:
                if track.is_drum:
                    track.program = (
                        0  # need to be done before sorting tracks per program
                    )

            # Sort and merge tracks if needed
            # MIDI produced with one_token_stream contains tracks with different orders
            # This step is also performed in preprocess_midi, but we need to call it here for the assertions below
            tokenizer.preprocess_midi(midi_to_compare)
            # For Octuple, as tempo is only carried at notes times, we need to adapt their times for comparison
            if tokenization in ["Octuple"]:
                adapt_tempo_changes_times(
                    midi_to_compare.instruments, midi_to_compare.tempo_changes
                )
            # When the tokenizer only decoded tempo changes different from the last tempo val
            if tokenization in ["CPWord"]:
                remove_equal_successive_tempos(midi_to_compare.tempo_changes)

            # MIDI -> Tokens -> MIDI
            midi_to_compare.instruments.sort(
                key=lambda x: (x.program, x.is_drum)
            )  # sort tracks
            tokens = tokenizer(midi_to_compare)
            new_midi = tokenizer(
                tokens,
                miditok.utils.get_midi_programs(midi_to_compare),
                time_division=midi_to_compare.ticks_per_beat,
            )
            new_midi.instruments.sort(key=lambda x: (x.program, x.is_drum))

            # Checks types and values conformity following the rules
            tokens_types = tokenizer.tokens_errors(
                tokens[0] if not tokenizer.one_token_stream else tokens
            )
            if tokens_types != 0.0:
                print(
                    f"Validation of tokens types / values successions failed with {tokenization}: {tokens_types:.2f}"
                )

            # Checks notes
            errors = midis_equals(midi_to_compare, new_midi)
            if len(errors) > 0:
                has_errors = True
                for e, track_err in enumerate(errors):
                    if track_err[-1][0][0] != "len":
                        for err, note, exp in track_err[-1]:
                            new_midi.markers.append(
                                Marker(
                                    f"{e}: with note {err} (pitch {note.pitch})",
                                    note.start,
                                )
                            )
                print(
                    f"MIDI {i} - {file_path.stem} / {tokenization} failed to encode/decode NOTES"
                    f"({sum(len(t[2]) for t in errors)} errors)"
                )

            # Checks tempos
            if (
                tokenizer.config.use_tempos and tokenization != "MuMIDI"
            ):  # MuMIDI doesn't decode tempos
                tempo_errors = tempo_changes_equals(
                    midi_to_compare.tempo_changes, new_midi.tempo_changes
                )
                if len(tempo_errors) > 0:
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path.stem} / {tokenization} failed to encode/decode TEMPO changes"
                        f"({len(tempo_errors)} errors)"
                    )

            # Checks time signatures
            if tokenizer.config.use_time_signatures:
                time_sig_errors = time_signature_changes_equals(
                    midi_to_compare.time_signature_changes,
                    new_midi.time_signature_changes,
                )
                if len(time_sig_errors) > 0:
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path.stem} / {tokenization} failed to encode/decode TIME SIGNATURE changes"
                        f"({len(time_sig_errors)} errors)"
                    )

            # Checks pedals
            if tokenizer.config.use_sustain_pedals:
                pedal_errors = pedal_equals(midi_to_compare, new_midi)
                if any(len(err) > 0 for err in pedal_errors):
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path.stem} / {tokenization} failed to encode/decode PEDALS"
                        f"({sum(len(err) for err in pedal_errors)} errors)"
                    )

            # Checks pitch bends
            if tokenizer.config.use_pitch_bends:
                pitch_bend_errors = pitch_bend_equals(midi_to_compare, new_midi)
                if any(len(err) > 0 for err in pitch_bend_errors):
                    has_errors = True
                    print(
                        f"MIDI {i} - {file_path.stem} / {tokenization} failed to encode/decode PITCH BENDS"
                        f"({sum(len(err) for err in pitch_bend_errors)} errors)"
                    )

            # TODO check control changes

            if has_errors:
                has_errors = True
                if saving_erroneous_midis:
                    new_midi.dump(
                        Path(
                            "tests",
                            "test_results",
                            f"{file_path.stem}_{tokenization}.mid",
                        )
                    )
                    midi_to_compare.dump(
                        Path(
                            "tests",
                            "test_results",
                            f"{file_path.stem}_{tokenization}_original.mid",
                        )
                    )

    ttotal = time() - t0
    print(f"Took {ttotal:.2f} seconds")
    assert not has_errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Encoding test")
    parser.add_argument(
        "--data",
        type=str,
        default="tests/Multitrack_MIDIs",
        help="directory of MIDI files to use for test",
    )
    args = parser.parse_args()

    test_multitrack_midi_to_tokens_to_midi(args.data)
