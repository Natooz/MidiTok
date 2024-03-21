=================================
Using MidiTok with PyTorch models
=================================

MidiTok features PyTorch ``Dataset`` objects to load MIDI or token files during training.
You can use them with a PyTorch ``DataLoader`` or your preferred libraries.
When indexed, the ``Dataset`` will output dictionaries with values corresponding to the inputs and labels.

MidiTok also provides an "all-in-one" data collator: :class:`miditok.pytorch_data.DataCollator` to be used with PyTorch a ``DataLoader`` in order to pad batches, add `BOS` and `EOS` tokens and create attention masks.

**Note:** *This module is imported only if* ``torch`` *is installed in your Python environment.*

.. automodule:: miditok.pytorch_data
    :members:
