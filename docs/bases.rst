=================
Bases
=================

This page features the bases of MidiTok, of how tokenizers work.

Tokens and vocabulary
------------------------

A token is a distinct element, part of a sequence of tokens. In natural language, a token can be a character, a subword or a word. A sentence can then be tokenized into a sequence of tokens representing the words and punctuation.
For symbolic music, tokens can represent the values of the note attributes (pitch, valocity, duration) or time events. These are the "basic" tokens, that can be compared to the characters in natural language. In the vocabulary of trained tokenizers, the tokens can represent **successions** of these basic tokens.
A token can take three forms, which we name by convention:

* Token (``string``): the form describing it, e.g. *Pitch_50*.
* Id (``int``): an unique associated integer, which corresponds to the index of the index in the vocabulary.
* Byte (``string``): an distinct byte, used internally for trained tokenizers (:ref:`Training a tokenizer`).

MidiTok works with :ref:`TokSequence` objects to conveniently represent these three forms.

Vocabulary
------------------------

The vocabulary of a tokenizer acts as a lookup table, linking tokens (string / byte) to their ids (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``. The vocabulary is a Python dictionary binding tokens (keys) to their ids (values).
For tokenizations with embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

**With Byte Pair Encoding:**
``tokenizer.vocab`` holds all the basic tokens describing the note and time attributes of music. By analogy with text, these tokens can be seen as unique characters.
After :ref:`Training a tokenizer`, a new vocabulary is built with newly created tokens from pairs of basic tokens. This vocabulary can be accessed with ``tokenizer.vocab_bpe``, and binds tokens as bytes (string) to their associated ids (int). This is the vocabulary of the 🤗tokenizers BPE model.

TokSequence
------------------------

The methods of MidiTok use :class:`miditok.TokSequence` objects as input and outputs. A :class:`miditok.TokSequence` holds tokens as the three forms described in :ref:`Byte Pair Encoding (BPE)`. TokSequences are subscriptable and implement ``__len__`` (you can run ``tok_seq[id]`` and ``len(tok_seq)``).

You can use the :py:func:`miditok.MusicTokenizer.complete_sequence` method to automatically fill the non-initialized attributes of a :class:`miditok.TokSequence`.

.. autoclass:: miditok.TokSequence
    :members:

The MusicTokenizer class
------------------------

MidiTok features several MIDI tokenizations, all inheriting from the :class:`miditok.MusicTokenizer` class.
You can customize your tokenizer by creating it with a custom :ref:`Tokenizer config`.

.. autoclass:: miditok.MusicTokenizer
    :members:

Tokenizer config
------------------------

All tokenizers are initialized with common parameters, that are hold in a :class:`miditok.TokenizerConfig` object, documented below. You can access a tokenizer's configuration with `tokenizer.config`.
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


Tokens & TokSequence input / output format
--------------------------------------------

Depending on the tokenizer at use, the **format** of the tokens returned by the :py:func:`miditok.MusicTokenizer.encode` method may vary, as well as the expected format for the :py:func:`miditok.MusicTokenizer.decode` method. The format is given by the :py:func:`miditok.MusicTokenizer.io_format` property. For any tokenizer, the format is the same for both methods.

The format is deduced from the :py:func:`miditok.MusicTokenizer.is_multi_voc` and ``one_token_stream`` tokenizer attributes. ``one_token_stream`` being ``True`` means that the tokenizer will convert a MIDI file into a single stream of tokens for all instrument tracks, otherwise it will convert each track to a distinct token sequence. :py:func:`miditok.MusicTokenizer.is_multi_voc` being True means that each token stream is a list of lists of tokens, of shape ``(T,C)`` for T time steps and C subtokens per time step.

This results in four situations, where I is the number of tracks, T is the number of tokens (or time steps) and C the number of subtokens per time step:

* ``is_multi_voc`` and ``one_token_stream`` are both ``False``: ``[I,(T)]``;
* ``is_multi_voc`` is ``False`` and ``one_token_stream`` is ``True``: ``(T)``;
* ``is_multi_voc`` is ``True`` and ``one_token_stream`` is ``False``: ``[I,(T,C)]``;
* ``is_multi_voc`` and ``one_token_stream`` are both ``True``: ``(T,C)``.

**Note that if there is no I dimension in the format, the output of** :py:func:`miditok.MusicTokenizer.encode` **is a** :class:`miditok.TokSequence` **object, otherwise it is a list of** :class:`miditok.TokSequence` **objects (one per token stream / track).**

Some tokenizer examples to illustrate:

* **TSD** without ``config.use_programs`` will not have multiple vocabularies and will treat each MIDI track as a unique stream of tokens, hence it will convert MIDI files to a list of :class:`miditok.TokSequence` objects, ``(I,T)`` format.
* **TSD** with ``config.use_programs`` being True will convert all MIDI tracks to a single stream of tokens, hence one :class:`miditok.TokSequence` object, ``(T)`` format.
* **CPWord** is a multi-voc tokenizer, without ``config.use_programs`` it will treat each MIDI track as a distinct stream of tokens, hence it will convert MIDI files to a list of :class:`miditok.TokSequence` objects with the ``(I,T,C)`` format.
* **Octuple** is a multi-voc tokenizer and converts all MIDI track to a single stream of tokens, hence it will convert MIDI files to a :class:`miditok.TokSequence` object, ``(T,C)`` format.


Magic methods
------------------------

`Magic methods <https://rszalski.github.io/magicmethods/>`_ allows to intuitively access to a tokenizer's attributes and methods. We list them here with some examples.

.. autofunction:: miditok.MusicTokenizer.__call__
    :noindex:
..  code-block:: python

    tokens = tokenizer(midi)
    midi2 = tokenizer(tokens)

.. autofunction:: miditok.MusicTokenizer.__getitem__
    :noindex:
..  code-block:: python

    pad_token = tokenizer["PAD_None"]

.. autofunction:: miditok.MusicTokenizer.__len__
    :noindex:
..  code-block:: python

    num_classes = len(tokenizer)
    num_classes_per_vocab = tokenizer.len  # applicable to tokenizer with embedding pooling, e.g. CPWord or Octuple

.. autofunction:: miditok.MusicTokenizer.__eq__
    :noindex:
..  code-block:: python

    if tokenizer1 == tokenizer2:
        print("The tokenizers have the same vocabulary and configurations!")

Save / Load tokenizer
------------------------

You can save and load a tokenizer's parameters and vocabulary. This is especially useful to track tokenized datasets, and to save tokenizers with vocabularies learned with :ref:`Byte Pair Encoding (BPE)`.

.. autofunction:: miditok.MusicTokenizer.save_params
    :noindex:

To load a tokenizer from saved parameters, just use the ``params`` argument when creating a it:

..  code-block:: python

    tokenizer = REMI(params=Path("to", "tokenizer.json"))
