=======================
Tokenizer Configuration
=======================

This page features the bases of MidiTok, of how tokenizers work.

Tokenizer config
------------------------

All tokenizers are initialized with common parameters, that are hold in a :class:`miditok.TokenizerConfig` object, documented below. You can access a tokenizer's configuration with ``tokenizer.config``.
Some tokenizers might take additional specific arguments / parameters when creating them.

.. autoclass:: miditok.TokenizerConfig
    :members:


Additional tokens
------------------------

MidiTok offers to include additional tokens on music information. You can specify them in the ``tokenizer_config`` argument (:class:`miditok.TokenizerConfig`) when creating a tokenizer. The :class:`miditok.TokenizerConfig` documentations specifically details the role of each of them, and their associated parameters.

.. csv-table:: Compatibility table of tokenizations and additional tokens.
   :file: additional_tokens_table.csv
   :header-rows: 1

¹: using both time signatures and rests with :class:`miditok.CPWord` might result in time alterations, as the time signature changes are carried with the Bar tokens which can be skipped during period of rests.
²: using time signatures with :class:`miditok.Octuple` might result in time alterations, as the time signature changes are carried with the note onsets. An example is shown below.

.. image:: /assets/Octuple_TS_Rest/original.png
  :width: 800
  :alt: Original MIDI sample preprocessed / downsampled

.. image:: /assets/Octuple_TS_Rest/tokenized.png
  :width: 800
  :alt: MIDI sample after being tokenized, the time has been shifted to a bar during the time signature change

Below is an example of how pitch intervals would be tokenized, with a ``max_pitch_interval`` of 15.

.. image:: /assets/pitch_intervals.png
  :width: 800
  :alt: Schema of the pitch intervals over a piano-roll


Special tokens
------------------------

MidiTok offers to include some special tokens to the vocabulary. These tokens with no "musical" information can be used for training purposes.
To use special tokens, you must specify them with the ``special_tokens`` argument when creating a tokenizer. By default, this argument is set to ``["PAD", "BOS", "EOS", "MASK"]``. Their signification are:

* **PAD** (``PAD_None``): a padding token to use when training a model with batches of sequences of unequal lengths. The padding token id is often set to 0. If you use Hugging Face models, be sure to pad inputs with this tokens, and pad labels with *-100*.
* **BOS** (``SOS_None``): "Start Of Sequence" token, indicating that a token sequence is beginning.
* **EOS** (``EOS_None``): "End Of Sequence" tokens, indicating that a token sequence is ending. For autoregressive generation, this token can be used to stop it.
* **MASK** (``MASK_None``): a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.

**Note:** you can use the ``tokenizer.special_tokens`` property to get the list of the special tokens of a tokenizer, and ``tokenizer.special_tokens`` for their ids.

