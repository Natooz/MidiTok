=================
Code examples
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

Here we convert a MIDI to tokens, decode them back to a MIDI.

..  code-block:: python

    from pathlib import Path

    # Tokenize a MIDI file
    tokens = tokenizer(Path("to", "your_midi.mid"))  # automatically detects Score objects, paths, tokens

    # Convert to MIDI and save it
    generated_midi = tokenizer(tokens)  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors
    generated_midi.dump_midi(Path("to", "decoded_midi.mid"))


Trains a tokenizer with BPE
-----------------------------

Here we train the tokenizer with :ref:`Byte Pair Encoding (BPE)`.
BPE allows to reduce the lengths of the sequences of tokens, in turn model efficiency, while improving the results quality/model performance.

..  code-block:: python

    from miditok import REMI
    from pathlib import Path

    # Creates the tokenizer and list the file paths
    tokenizer = REMI()  # using defaults parameters (constants.py)
    midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))

    # Builds the vocabulary with BPE
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)


Prepare a dataset before training
-------------------------------------------

MidiTok provides useful methods to split music files into smaller chunks that make approximately a target number of tokens, allowing to use most of your data to train and evaluate models. It also provide data augmentation methods to increase the amount of data to train models.

..  code-block:: python

    from random import shuffle

    from miditok.data_augmentation import augment_dataset
    from miditok.utils import split_files_for_training

    # Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
    midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))
    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    # Chunk MIDIs and perform data augmentation on each subset independently
    for files_paths, subset_name in (
        (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):

        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"dataset_{subset_name}")
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=1024,
            num_overlap_bars=2,
        )

        # Perform data augmentation
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-12, 12],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5],
        )

Creates a Dataset and collator for training
-------------------------------------------

Creates a Dataset and a collator to be used with a PyTorch DataLoader to train a model

..  code-block:: python

    from miditok import REMI
    from miditok.pytorch_data import DatasetMIDI, DataCollator
    from torch.utils.data import DataLoader

    tokenizer = REMI()  # using defaults parameters (constants.py)
    midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))
    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator)

    # Using the data loader in the training loop
    for batch in data_loader:
        print("Train your model on this batch...")


Tokenize a dataset
------------------------

Here we tokenize a whole dataset into JSON files storing the tokens ids.
We also perform data augmentation on the pitch, velocity and duration dimension.

..  code-block:: python

    from miditok import REMI
    from miditok.data_augmentation import augment_midi_dataset
    from pathlib import Path

    # Creates the tokenizer and list the file paths
    tokenizer = REMI()  # using defaults parameters (constants.py)
    data_path = Path("path", "to", "dataset")

    # A validation method to discard MIDIs we do not want
    # It can also be used for custom pre-processing, for instance if you want to merge
    # some tracks before tokenizing a MIDI file
    def midi_valid(midi) -> bool:
        if any(ts.numerator != 4 for ts in midi.time_signature_changes):
            return False  # time signature different from 4/*, 4 beats per bar
        return True

    # Performs data augmentation on one pitch octave (up and down), velocities and
    # durations
    midi_aug_path = Path("to", "new", "location", "augmented")
    augment_midi_dataset(
        data_path,
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 5],
        duration_offsets=[-0.5, 1],
        out_path=midi_aug_path,
    )
    tokenizer.tokenize_dataset(        # 2 velocity and 1 duration values
        data_path,
        Path("path", "to", "tokens"),
        midi_valid,
    )
