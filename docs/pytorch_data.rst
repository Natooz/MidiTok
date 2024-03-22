=================================
PyTorch Training
=================================

MidiTok features PyTorch `Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ objects to data during training, usually coupled with a PyTorch ``DataLoader``. A ``Dataset`` is an object storing the information about a dataset: paths of files to load, or the data itself stored in memory (recommended for small datasets only).
When indexed, the ``Dataset`` will output dictionaries with values corresponding to the inputs and labels.

For most purposes, the :class:`miditok.pytorch_data.DatasetMIDI` should fulfill your needs. And don't forget to use the :py:func:`miditok.pytorch_data.split_midis_for_training` method to split your MIDI files into chunks sizes optimized for your desired token sequence length.

MidiTok also provides an "all-in-one" data collator: :class:`miditok.pytorch_data.DataCollator` to be used with PyTorch a ``DataLoader`` in order to pad batches and create attention masks.

**Note:** This module is imported only if ``torch`` is installed in your Python environment.

.. automodule:: miditok.pytorch_data
    :members:
