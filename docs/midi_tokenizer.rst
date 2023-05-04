=================
Basics
=================

This page features the bases of MidiTok, of how tokenizers work.

Tokens and vocabulary
------------------------

A token can take three forms, which we name by convention:

* Token (``string``): the form describing it, e.g. *Pitch_50*.
* Id (``int``): an unique associated integer, used as an index.
* Byte (``string``): an unique associated byte, used internally for :ref:`Byte Pair Encoding (BPE)`.

MidiTok works with :ref:`TokSequence` objects to output token sequences of represented by these three forms.

TokSequence
------------------------

The methods of MidiTok use :class:`miditok.TokSequence` objects as input and outputs. A ``TokSequence`` holds tokens as the three forms described in :ref:`Byte Pair Encoding (BPE)`. TokSequences are subscriptable and implement ``__len__`` (you can run ``tok_seq[id]`` and ``len(tok_seq)``).

You can use the :py:func:`miditok.MIDITokenizer.complete_sequence` method to automatically fill the non-initialized attributes of a ``TokSequence``.

.. autoclass:: miditok.TokSequence
    :noindex:
    :members:

Vocabulary
------------------------

The vocabulary of a tokenizer acts as a lookup table, linking tokens (string) to their ids (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``. The vocabulary is a Python dictionary binding tokens (keys) to their ids (values).
For tokenizations with embedding embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

**With Byte Pair Encoding:**
``tokenizer.vocab`` holds all the basic tokens describing the note and time attributes of music. By analogy with text, these tokens can be seen as unique characters.
After training a tokenizer with :ref:`Byte Pair Encoding (BPE)`, a new vocabulary is built with newly created tokens from pairs of basic tokens. This vocabulary can be accessed with ``tokenizer.vocab_bpe``, and binds tokens as bytes (string) to their associated ids (int). This is the vocabulary of the 🤗tokenizers BPE model.

MIDI Tokenizer
------------------------

MidiTok features several MIDI tokenizations, all inheriting from the :class:`miditok.MIDITokenizer` class.
The documentation of the arguments teaches you how to create a custom tokenizer.

.. autoclass:: miditok.MIDITokenizer
    :members:

Additional tokens
------------------------

MidiTok offers to include additional tokens on music information. You can specify them in the ``additional_tokens`` argument when creating a tokenizer.

* **Chords:** indicates the presence of a chord at a certain time step. MidiTok uses a chord detection method based on onset times and duration. This allows MidiTok to detect precisely chords without ambiguity, whereas most chord detection methods in symbolic music based on chroma features can't.
* **Rests:** includes *Rest* tokens whenever a portion of time is silent, i.e. no note is being played. This token type is decoded as a *TimeShift* event. You can choose the minimum and maximum rests values to represent with the ``rest_range`` key in the ``additional_tokens`` dictionary (default is 1/2 beat to 8 beats). Note that rests shorter than one beat are only divisible by the first beat resolution, e.g. a rest of 5/8th of a beat will be a succession of ``Rest_0.4`` and ``Rest_0.1``, where the first number indicate the rest duration in beats and the second in samples / positions.
* **Tempos:** specifies the current tempo. This allows to train a model to predict tempo changes. Tempo values are quantized accordingly to the ``nb_tempos`` and ``tempo_range`` entries in the ``additional_tokens`` dictionary (default is 32 tempos from 40 to 250).
* **Programs:** used to specify an instrument / MIDI program. MidiTok only offers the possibility to include these tokens in the vocabulary for you, but won't use them. If you need model multitrack symbolic music with other methods than Octuple / MuMIDI, MidiTok leaves you the choice / task to represent the track information the way you want. You can do it as in `LakhNES <https://github.com/chrisdonahue/LakhNES>`_ or `MMM <https://metacreation.net/mmm-multi-track-music-machine/>`_.
* **Time Signature:** specifies the current time signature. Only implemented with :ref:`REMIPlus`, :ref:`Octuple` and :ref:`Octuple Mono` atow.

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
   * - Chord
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
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
   * - Tempo
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
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
   * - Time signature
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌

Special tokens
------------------------

MidiTok offers to include some special tokens to the vocabulary. These tokens with no "musical" information can be used for training purposes.
To use special tokens, you must specify them with the ``special_tokens`` argument when creating a tokenizer. By default, this argument is set to ``["PAD", "BOS", "EOS", "MASK"]``. Their signification are:

* **PAD** (``PAD_None``): a padding token to use when training a model with batches of sequences of unequal lengths. The padding token id is often set to 0. If you use Hugging Face models, be sure to pad inputs with this tokens, and pad labels with *-100*.
* **BOS** (``SOS_None``): "Start Of Sequence" token, indicating that a token sequence is beginning.
* **EOS** (``EOS_None``): "End Of Sequence" tokens, indicating that a token sequence is ending. For autoregressive generation, this token can be used to stop it.
* **MASK** (``MASK_None``): a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.

**Note:** you can use the ``tokenizer.special_tokens`` argument to get the list of the special tokens of a tokenizer.

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
        print("The tokenizers have the same vocabulary!")

Save / Load tokenizer
------------------------

You can save and load a tokenizer's parameters and vocabulary. This is especially useful to track tokenized datasets, and to save tokenizers with vocabularies learned with :ref:`Byte Pair Encoding (BPE)`.

.. autofunction:: miditok.MIDITokenizer.save_params
    :noindex:
.. autofunction:: miditok.MIDITokenizer.load_params
    :noindex:

Limitations
------------------------

Some tokenizations using Bar tokens (:ref:`REMI`, :ref:`CPWord` and :ref:`MuMIDI`) only considers a 4/x time signature for now. This means that each bar is considered covering 4 beats.
:ref:`REMIPlus` and :ref:`Octuple` supports it.
