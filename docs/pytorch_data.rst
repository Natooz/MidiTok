=================================
Using MidiTok with Pytorch
=================================

MidiTok features PyTorch `Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ objects to load music data during training, usually coupled with a PyTorch ``DataLoader``. A ``Dataset`` is an object storing the information about a dataset: paths of files to load, or the data itself stored in memory (recommended for small datasets only).
When indexed, the ``Dataset`` will output dictionaries with values corresponding to the inputs and labels.

Loading data
--------------------------

MidiTok provides two dataset classes: :class:`miditok.pytorch_data.DatasetMIDI` and :class:`miditok.pytorch_data.DatasetJSON`.

:class:`miditok.pytorch_data.DatasetMIDI` loads MIDI files and can either tokenize them on the fly when the dataset is indexed, or pre-tokenize them when creating it and saving the token ids in memory. **For most use cases, this Dataset should fulfill your needs and is recommended.**

:class:`miditok.pytorch_data.DatasetJSON` loads JSON files containing token ids. It requires to first tokenize a dataset to be used. This dataset is only compatible with JSON files saved as "one token stream" (``tokenizer.one_token_stream``). In order to use it for all the tracks of a multi-stream tokenizer, you will need to save each track token sequence as a separate JSON file.

Preparing data
--------------------------

When training a model, you will likely want to limit the possible token sequence length in order to not run out of memory. The dataset classes handle such case and can trim the token sequences. However, **it is not uncommon for a single MIDI to be tokenized into sequences that can contain several thousands tokens, depending on its duration and number of notes. In such case, using only the first portion of the token sequence would considerably reduce the amount of data used to train and test a model.**

To handle such case, MidiTok provides the :py:func:`miditok.pytorch_data.split_files_for_training` method to dynamically split MIDI files into chunks that should be tokenized in approximately the number of tokens you want.
If you cannot fit most of your MIDIs into single usable token sequences, we recommend to split your dataset with this method.

Data loading example
--------------------------

MidiTok also provides an "all-in-one" data collator: :class:`miditok.pytorch_data.DataCollator` to be used with PyTorch a ``DataLoader`` in order to pad batches and create attention masks.
Here is a complete example showing how to use this module to train any model.

..  code-block:: python

    from miditok import REMI, TokenizerConfig
    from miditok.pytorch_data import DatasetMIDI, DataCollator, split_files_for_training
    from torch.utils.data import DataLoader
    from pathlib import Path

    # Creating a multitrack tokenizer configuration, read the doc to explore other parameters
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)

    # Train the tokenizer with Byte Pair Encoding (BPE)
    midi_paths = list(Path("path", "to", "midis").glob("**/*.mid"))
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    tokenizer.save_params(Path("path", "to", "save", "tokenizer.json"))
    # And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
    tokenizer.push_to_hub("username/model-name", private=True, token="your_hf_token")

    # Split MIDIs into smaller chunks for training
    dataset_chunks_dir = Path("path", "to", "midi_chunks")
    split_files_for_training(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        save_dir=dataset_chunks_dir,
        max_seq_len=1024,
    )

    # Create a Dataset, a DataLoader and a collator to train a model
    dataset = DatasetMIDI(
        files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

    # Iterate over the dataloader to train a model
    for batch in dataloader:
        print("Train your model on this batch...")

**Note:** This module is imported only if ``torch`` is installed in your Python environment.

.. automodule:: miditok.pytorch_data
    :members:
