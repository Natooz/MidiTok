=================
MIDI Tokenizer
=================

MidiTok features several MIDI tokenizations, all inheriting from a ``MIDITokenizer`` class. We recommend you to explore its methods.

.. autoclass:: miditok.MIDITokenizer
    :members:

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

