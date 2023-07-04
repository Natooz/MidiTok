=================
Examples
=================

Create a tokenizer
------------------------

A basic example showing how to create a tokenizer, with a selection of custom parameters.

..  code-block:: python

    from miditok import REMI, TokenizerConfig  # here we choose to use REMI
    from miditok.utils import get_midi_programs

    # Our parameters
    TOKENIZER_PARAMS = {
        "pitch_range": range(21, 109),
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

    # Creates the tokenizer and loads a MIDI
    tokenizer = REMI(config)

MIDI - Tokens conversion
-------------------------------

Here we convert a MIDI to tokens, and the other way around.

..  code-block:: python

    from miditok.utils import get_midi_programs
    from miditoolkit import MidiFile

    # Tokenize a MIDI file
    midi_path = "path/to/your_midi.mid"
    midi = MidiFile(midi_path)
    tokens = tokenizer(midi_path)  # automatically detects MidiFile, paths or tokens before converting them

    # Convert to MIDI and save it
    programs = get_midi_programs(midi)
    generated_midi = tokenizer(tokens, programs=programs)  # MidiTok can handle PyTorch / Tensorflow Tensors
    generated_midi.dump('path/to/save/file.mid')  # could have been done above by giving the path argument

Tokenize a dataset
------------------------

Here we tokenize a whole dataset.
We also perform data augmentation on the pitch, velocity and duration dimension.
Finally, we learn :ref:`Byte Pair Encoding (BPE)` on the tokenized dataset, and apply it.

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
        if midi.max_tick < 10 * midi.ticks_per_beat:
            return False  # this MIDI is too short
        return True

    # Converts MIDI files to tokens saved as JSON files
    data_augmentation_offsets = [2, 2, 1]   # will perform data augmentation on 2 pitch octaves,
    tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
        midi_paths,
        Path('path', 'to', 'tokens_noBPE'),
        midi_valid,
        data_augmentation_offsets,
    )

    # Learns the vocabulary with BPE
    tokenizer.learn_bpe(
        vocab_size=500,
        tokens_paths=list(Path('path', 'to', 'tokens_noBPE').glob("**/*.json")),
        out_dir=Path('path', 'to', 'tokens_BPE'),
    )

    # Converts the tokenized musics into tokens with BPE
    tokenizer.apply_bpe_to_dataset(Path('path', 'to', 'tokens_noBPE'), Path('path', 'to', 'tokens_BPE'))

