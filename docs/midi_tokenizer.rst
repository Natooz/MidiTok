MIDI Tokenizer
=====

Base tokenizer class
----------------

MidiTok features several MIDI tokenizations, that are all built from a common ``MIDITokenizer`` class.

To retrieve a list of random ingredients,
you can use the ``miditok.MIDITokenizer()`` function:

.. autofunction:: miditok.MIDITokenizer.token_types_errors

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`miditok.MIDITokenizer.token_types_errors`
will raise an exception.

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

Vocabulary
----------------

Tokenize dataset
----------------

Magic methods
----------------

call, len, get...

Save / Load tokenizer
----------------
