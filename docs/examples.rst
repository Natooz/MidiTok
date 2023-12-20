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
        "nb_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "nb_tempos": 32,  # nb of tempo bins
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
    generated_midi.dump_midi('path/to/save/file.mid')  # could have been done above by giving the path argument

Tokenize a dataset
------------------------

Here we first train the tokenizer with :ref:`Byte Pair Encoding (BPE)`, then we tokenize a whole dataset.
We also perform data augmentation on the pitch, velocity and duration dimension.

..  code-block:: python

    from miditok import REMI
    from pathlib import Path

    # Creates the tokenizer and list the file paths
    tokenizer = REMI()  # using defaults parameters (constants.py)
    midi_paths = list(Path('path', 'to', 'dataset').glob('**/*.mid'))

    # A validation method to discard MIDIs we do not want
    # It can also be used for custom pre-processing, for instance if you want to merge
    # some tracks before tokenizing a MIDI file
    def midi_valid(midi) -> bool:
        if any(ts.numerator != 4 for ts in midi.time_signature_changes):
            return False  # time signature different from 4/*, 4 beats per bar
        return True

    # Learns the vocabulary with BPE
    tokenizer.learn_bpe(vocab_size=10000, files_paths=midi_paths)

    # Converts MIDI files to tokens saved as JSON files
    data_augmentation_offsets = [2, 2, 1]   # will perform data augmentation on 2 pitch octaves,
    tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
        midi_paths,
        Path('path', 'to', 'tokens_noBPE'),
        midi_valid,
        data_augmentation_offsets,
    )
