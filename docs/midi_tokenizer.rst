=================
MIDI Tokenizer
=================

MidiTok features several MIDI tokenizations, all inheriting from a ``MIDITokenizer`` class.
Read the documentation of the arguments of :class:`miditok.MIDITokenizer` to learn how to

.. autoclass:: miditok.MIDITokenizer
    :members:

Additional tokens
------------------------

MidiTok offers to include additional tokens on music information. You can specify them in the ``additional_tokens`` argument when creating a tokenizer.

* **Chords:** indicates the presence of a chord at a certain time step. MidiTok uses a chord detection method based on onset times and duration. This allows MidiTok to detect precisely chords without ambiguity, whereas most chord detection methods in symbolic music based on chroma features can't.
* **Rests:** includes *Rest* tokens whenever a portion of time is silent, i.e. no note is being played. This token type is decoded as a *TimeShift* event. You can choose the minimum and maximum rests values to represent with the ``rest_range`` key in the ``additional_tokens`` dictionary (default is 1/2 beat to 8 beats). Note that rests shorter than one beat are only divisible by the first beat resolution, e.g. a rest of 5/8th of a beat will be a succession of ``Rest_0.4`` and ``Rest_0.1``, where the first number indicate the rest duration in beats and the second in samples / positions.
* **Tempos:** specifies the current tempo. This allows to train a model to predict tempo changes. Tempo values are quantized accordingly to the ``nb_tempos`` and ``tempo_range`` entries in the ``additional_tokens`` dictionary (default is 32 tempos from 40 to 250).
* **Programs:** used to specify an instrument / MIDI program. MidiTok only offers the possibility to include these tokens in the vocabulary for you, but won't use them. If you need model multitrack symbolic music with other methods than Octuple / MuMIDI, MidiTok leaves you the choice / task to represent the track information the way you want. You can do it as in `LakhNES <https://github.com/chrisdonahue/LakhNES>`_ or `MMM <https://metacreation.net/mmm-multi-track-music-machine/>`_.
* **Time Signature:** specifies the current time signature. Only implemented with :ref:`Octuple` in MidiTok a.t.w.

.. list-table:: Compatibility table of tokenizations and additional tokens.
   :header-rows: 1

   * - Token type
     - :ref:`REMI`
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
     - ❌
     - ❌
     - ✅
   * - Rest
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
   * - Time signature
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌

Special tokens
------------------------

MidiTok offers to include some special tokens to the vocabulary. To use them, you must specify them when creating a tokenizer (constructor argument). These are:

* **pad** (default ``True``) --> ``PAD_None``: a padding token to use when training a model with batches of sequences of unequal lengths. The padding token will be at index 0 of the vocabulary.
* **sos_eos** (default ``False``) --> ``SOS_None`` and ``EOS_None``: "Start Of Sequence" and "End Of Sequence" tokens, designed to be placed respectively at the beginning and end of a token sequence during training. At inference, the EOS token tells when to end the generation.
* **mask** (default ``False``) --> ``MASK_None``: a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.
* **sep** (default: ``False``) --> ``SEP_None``: a token to use as a separation between sequences.

**Note:** you can use the ``tokenizer.special_tokens`` property to get the list of the special tokens of a tokenizer.

Vocabulary
------------------------

The ``Vocabulary`` class acts as a lookup table, linking tokens (*Pitch*...) to their index (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``.
For tokenizations with embedding embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

.. autoclass:: miditok.Vocabulary
    :noindex:
    :members:

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

Tokenizations using Bar tokens (:ref:`REMI`, :ref:`CPWord` and :ref:`MuMIDI`) only considers a 4/x time signature for now. This means that each bar is considered covering 4 beats.
:ref:`Octuple` supports it.
