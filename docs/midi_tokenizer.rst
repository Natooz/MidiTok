=================
Basics
=================

This page features the bases of MidiTok, of how tokenizers work.

Tokens and vocabulary
------------------------

A token is a distinct element, part of a sequence of tokens. In natural language, a token can be a character, a subword or a word. A sentence can then be tokenized into a sequence of tokens representing the words and punctuation.
For symbolic music, tokens can represent the values of the note attributes (pitch, valocity, duration) or time events. These are the "basic" tokens, that can be compared to the characters in natural language. With :ref:`Byte Pair Encoding (BPE)`, tokens can represent **successions** of these basic tokens.
A token can take three forms, which we name by convention:

* Token (``string``): the form describing it, e.g. *Pitch_50*.
* Id (``int``): an unique associated integer, used as an index.
* Byte (``string``): an unique associated byte, used internally for :ref:`Byte Pair Encoding (BPE)`.

MidiTok works with :ref:`TokSequence` objects to output token sequences of represented by these three forms.

Vocabulary
------------------------

The vocabulary of a tokenizer acts as a lookup table, linking tokens (string / byte) to their ids (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``. The vocabulary is a Python dictionary binding tokens (keys) to their ids (values).
For tokenizations with embedding embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

**With Byte Pair Encoding:**
``tokenizer.vocab`` holds all the basic tokens describing the note and time attributes of music. By analogy with text, these tokens can be seen as unique characters.
After training a tokenizer with :ref:`Byte Pair Encoding (BPE)`, a new vocabulary is built with newly created tokens from pairs of basic tokens. This vocabulary can be accessed with ``tokenizer.vocab_bpe``, and binds tokens as bytes (string) to their associated ids (int). This is the vocabulary of the 🤗tokenizers BPE model.

TokSequence
------------------------

The methods of MidiTok use :class:`miditok.TokSequence` objects as input and outputs. A ``TokSequence`` holds tokens as the three forms described in :ref:`Byte Pair Encoding (BPE)`. TokSequences are subscriptable and implement ``__len__`` (you can run ``tok_seq[id]`` and ``len(tok_seq)``).

You can use the :py:func:`miditok.MIDITokenizer.complete_sequence` method to automatically fill the non-initialized attributes of a ``TokSequence``.

.. autoclass:: miditok.TokSequence
    :noindex:
    :members:

MIDI Tokenizer
------------------------

MidiTok features several MIDI tokenizations, all inheriting from the :class:`miditok.MIDITokenizer` class.
You can customize your tokenizer by creating it with a custom :ref:`Tokenizer config`.

.. autoclass:: miditok.MIDITokenizer
    :members:

Tokenizer config
------------------------

All tokenizers are initialized with common parameters, that are hold in a ``TokenizerConfig`` object, documented below. You can access a tokenizer's configuration with `tokenizer.config`.
Some tokenizers might take additional specific arguments / parameters when creating them.

.. autoclass:: miditok.TokenizerConfig
    :members:

Additional tokens
------------------------

MidiTok offers to include additional tokens on music information. You can specify them in the ``tokenizer_config`` argument (:class:`miditok.TokenizerConfig`) when creating a tokenizer. The :class:`miditok.TokenizerConfig` documentations specifically details the role of each of them, and their associated parameters.

.. list-table:: Compatibility table of tokenizations and additional tokens.
   :header-rows: 1

   * - Token type
     - :ref:`REMI`
     - :ref:`REMIPlus`
     - :ref:`MIDI-Like`
     - :ref:`TSD`
     - :ref:`Structured`
     - :ref:`CPWord`
     - :ref:`Octuple`
     - :ref:`MuMIDI`
     - :ref:`MMM`
   * - Chord
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
   * - Rest
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Tempo
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
     - ✅
     - ✅
   * - Program
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Time signature
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ✅

Special tokens
------------------------

MidiTok offers to include some special tokens to the vocabulary. These tokens with no "musical" information can be used for training purposes.
To use special tokens, you must specify them with the ``special_tokens`` argument when creating a tokenizer. By default, this argument is set to ``["PAD", "BOS", "EOS", "MASK"]``. Their signification are:

* **PAD** (``PAD_None``): a padding token to use when training a model with batches of sequences of unequal lengths. The padding token id is often set to 0. If you use Hugging Face models, be sure to pad inputs with this tokens, and pad labels with *-100*.
* **BOS** (``SOS_None``): "Start Of Sequence" token, indicating that a token sequence is beginning.
* **EOS** (``EOS_None``): "End Of Sequence" tokens, indicating that a token sequence is ending. For autoregressive generation, this token can be used to stop it.
* **MASK** (``MASK_None``): a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.

**Note:** you can use the ``tokenizer.special_tokens`` property to get the list of the special tokens of a tokenizer, and ``tokenizer.special_tokens`` for their ids.

Magic methods
------------------------

`Magic methods <https://rszalski.github.io/magicmethods/>`_ allows to intuitively access to a tokenizer's attributes and methods. We list them here with some examples.

.. autofunction:: miditok.MIDITokenizer.__call__
    :noindex:
..  code-block:: python

    tokens = tokenizer(midi)
    midi2 = tokenizer(tokens)

.. autofunction:: miditok.MIDITokenizer.__getitem__
    :noindex:
..  code-block:: python

    pad_token = tokenizer["PAD_None"]

.. autofunction:: miditok.MIDITokenizer.__len__
    :noindex:
..  code-block:: python

    nb_classes = len(tokenizer)
    nb_classes_per_vocab = tokenizer.len  # applicable to tokenizer with embedding pooling, e.g. CPWord or Octuple

.. autofunction:: miditok.MIDITokenizer.__eq__
    :noindex:
..  code-block:: python

    if tokenizer1 == tokenizer2:
        print("The tokenizers have the same vocabulary and configurations!")

Save / Load tokenizer
------------------------

You can save and load a tokenizer's parameters and vocabulary. This is especially useful to track tokenized datasets, and to save tokenizers with vocabularies learned with :ref:`Byte Pair Encoding (BPE)`.

.. autofunction:: miditok.MIDITokenizer.save_params
    :noindex:

To load a tokenizer from saved parameters, just use the ``params`` argument when creating a it:

..  code-block:: python

    tokenizer = REMI(params=Path("to", "params.json"))

Limitations
------------------------

Some tokenizations using Bar tokens (:ref:`REMI`, :ref:`CPWord` and :ref:`MuMIDI`) only considers a 4/x time signature for now. This means that each bar is considered covering 4 beats.
:ref:`REMIPlus` and :ref:`Octuple` supports it.
