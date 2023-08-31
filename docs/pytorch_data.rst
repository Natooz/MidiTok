========================
PyTorch data loaders
========================

MidiTok features PyTorch `Dataset` objects to load MIDI or token files during training.
You can use them with PyTorch `DataLoader`s or your preferred libraries.
When indexed, the `Dataset`s will output dictionaries with values corresponding to the inputs and labels.

MidiTok also provides an "all-in-one" data collator: :class:`miditok.pytorch_data.DataCollator` to be used with PyTorch `DataLoader`s in order to pad batches, add `BOS` and `EOS` tokens and create attention masks.

**Note:** *This module is imported only if* `torch` *is installed in your Python environment.*

.. automodule:: miditok.data_augmentation
    :members:
