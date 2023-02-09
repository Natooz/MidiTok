=================
MIDI Tokenizer
=================

MidiTok features several MIDI tokenizations, all inheriting from a ``MIDITokenizer`` class.
Read the documentation of the arguments of :class:`miditok.MIDITokenizer` to learn how to

.. autoclass:: miditok.MIDITokenizer
    :members:

Additional tokens
------------------------

* **Chords:** indicates the presence of a chord at a certain time step. MidiTok uses a chord detection method based on onset times and duration. This allows MidiTok to detect precisely chords without ambiguity, whereas most chord detection methods in symbolic music based on chroma features can't.
* **Rests:** includes "Rest" events whenever a segment of time is silent, i.e. no note is played within. This token type is decoded as a "TimeShift" event, meaning the time will be shifted according to its value. You can choose the minimum and maximum rests values to represent (default is 1/2 beat to 8 beats). Note that rests shorter than one beat are only divisible by the first beat resolution, e.g. a rest of 5/8th of a beat will be a succession of ```Rest_0.4``` and ```Rest_0.1```, where the first number indicate the rest duration in beats and the second in samples / positions.
* **Tempos:** specifies the current tempo. This allows to train a model to predict tempo changes alongside with the notes, unless specified in the chart below. Tempo values are quantized on your specified range and number (default is 32 tempos from 40 to 250).
* **Programs:** used to specify an instrument / MIDI program. MidiTok only offers the possibility to include these tokens in the vocabulary for you, but won't use them. If you need model multitrack symbolic music with other methods than Octuple / MuMIDI, MidiTok leaves you the choice / task to represent the track information the way you want. You can do it as in [LakhNES](https://github.com/chrisdonahue/LakhNES) or [MMM](https://metacreation.net/mmm-multi-track-music-machine/).
* **Time Signature:** specifies the current time signature. Only implemented with Octuple in MidiTok a.t.w.

Special tokens
------------------------

* **`pad`** (default `True`) --> `PAD_None`: a padding token to use when training a model with batches of sequences of unequal lengths. The padding token will be at index 0 of the vocabulary.
* **`sos_eos`** (default `False`) --> `SOS_None` and `EOS_None`: "Start Of Sequence" and "End Of Sequence" tokens, designed to be placed respectively at the beginning and end of a token sequence during training. At inference, the EOS token tells when to end the generation.
* **`mask`** (default `False`) --> `MASK_None`: a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.
* **`sep`** (default: `False`) --> `SEP_None`: a token to use as a separation between sequences.

Vocabulary
------------------------

The ``Vocabulary`` class acts as a lookup table, linking tokens (*Pitch*...) to their index (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``.
For tokenizations with embedding embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

.. autoclass:: miditok.Vocabulary
    :noindex:
    :members:

Magic methods
------------------------

call, len, get...

Save / Load tokenizer
------------------------

Save / load

.. autofunction:: miditok.MIDITokenizer.save_params
    :noindex:
.. autofunction:: miditok.MIDITokenizer.load_params
    :noindex:

