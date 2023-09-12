#!/usr/bin/python3 python

"""Multitrack test file
"""

from copy import deepcopy
from pathlib import Path
from typing import Union
from time import time

import miditok
from miditoolkit import MidiFile, Pedal
from tqdm import tqdm

from .tests_utils import (
    ALL_TOKENIZATIONS,
    tokenize_check_equals,
    adapt_tempo_changes_times,
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
    "one_token_stream_for_programs": True,
    "program_changes": False,
}

# Define kwargs sets
# The first set is empty, using the default params
params_kwargs_sets = {tok: [{}] for tok in ALL_TOKENIZATIONS}
programs_tokenizations = ["TSD", "REMI", "MIDILike", "Structured", "CPWord", "Octuple"]
for tok in programs_tokenizations:
    params_kwargs_sets[tok].append(
        {"one_token_stream_for_programs": False},
    )
for tok in ["TSD", "REMI", "MIDILike"]:
    params_kwargs_sets[tok].append(
        {"program_changes": True},
    )
# Disable tempos for Octuple with one_token_stream_for_programs, as tempos are carried by note tokens
params_kwargs_sets["Octuple"][1]["use_tempos"] = False
# Increase the TimeShift voc for Structured as it doesn't support successive TimeShifts
for kwargs_set in params_kwargs_sets["Structured"]:
    kwargs_set["beat_res"] = {(0, 512): 8}


def test_multitrack_midi_to_tokens_to_midi(
    data_path: Union[str, Path] = "./tests/Multitrack_MIDIs",
    saving_erroneous_midis: bool = False,
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed
    """
    files = list(Path(data_path).glob("**/*.mid"))
    at_least_one_error = False
    t0 = time()

    for fi, file_path in enumerate(tqdm(files, desc="Testing multitrack")):
        # Reads the MIDI
        midi = MidiFile(Path(file_path))
        if midi.ticks_per_beat % max(BEAT_RES_TEST.values()) != 0:
            continue
        # add pedal messages
        for ti in range(max(3, len(midi.instruments))):
            midi.instruments[ti].pedals = [
                Pedal(start, start + 200) for start in [100, 600, 1800, 2200]
            ]

        for tokenization in ALL_TOKENIZATIONS:
            for pi, params_kwargs in enumerate(params_kwargs_sets[tokenization]):
                idx = f"{fi}_{pi}"
                params = deepcopy(TOKENIZER_PARAMS)
                params.update(params_kwargs)
                tokenizer_config = miditok.TokenizerConfig(**params)
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
                decoded_midi, has_errors = tokenize_check_equals(
                    midi_to_compare, tokenizer, idx, file_path.stem
                )

                if has_errors:
                    at_least_one_error = True
                    if saving_erroneous_midis:
                        decoded_midi.dump(
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
    assert not at_least_one_error


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
