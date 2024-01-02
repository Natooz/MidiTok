=================
Examples
=================

Create a tokenizer
------------------------

A basic example showing how to create a tokenizer, with a selection of custom parameters.

..  code-block:: python

    from miditok import REMI, TokenizerConfig  # here we choose to use REMI

    # Our parameters
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)

    # Creates the tokenizer
    tokenizer = REMI(config)

MIDI - Tokens conversion
-------------------------------

Here we convert a MIDI to tokens, and the other way around.

..  code-block:: python

    from miditoolkit import MidiFile

    # Tokenize a MIDI file
    midi = MidiFile("path/to/your_midi.mid")
    tokens = tokenizer(midi_path)  # automatically detects MidiFile, paths

    # Convert to MIDI and save it
    generated_midi = tokenizer(tokens)  # MidiTok can handle PyTorch / Tensorflow Tensors
    generated_midi.dump_midi("path/to/save/file.mid")  # could have been done above by giving the path argument

Tokenize a dataset
------------------------

Here we first train the tokenizer with :ref:`Byte Pair Encoding (BPE)`, then we tokenize a whole dataset.
We also perform data augmentation on the pitch, velocity and duration dimension.

..  code-block:: python

    from miditok import REMI
    from miditok.data_augmentation import augment_midi_dataset
    from pathlib import Path

    # Creates the tokenizer and list the file paths
    tokenizer = REMI()  # using defaults parameters (constants.py)
    midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))

    # A validation method to discard MIDIs we do not want
    # It can also be used for custom pre-processing, for instance if you want to merge
    # some tracks before tokenizing a MIDI file
    def midi_valid(midi) -> bool:
        if any(ts.numerator != 4 for ts in midi.time_signature_changes):
            return False  # time signature different from 4/*, 4 beats per bar
        return True

    # Learns the vocabulary with BPE
    tokenizer.learn_bpe(vocab_size=30000, files_paths=midi_paths)

    # Performs data augmentation on one pitch octave (up and down), velocities and
    # durations
    augment_midi_dataset(
        midi_paths,
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 5],
        duration_offsets=[-0.5, 1],
        out_path=midi_aug_path,
        Path("to", "new", "location", "augmented"),
    )
    tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
        midi_paths,
        Path("path", "to", "tokens"),
        midi_valid,
    )
