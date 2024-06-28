=================
Bases of MidiTok
=================

This page introduces the bases of MidiTok, how a tokenizer works and what are the basic elements of MidiTok.

MidiTok's workflow
------------------------

MidiTok uses a common workflow for all its tokenizers, which follows:

1. **Music file preprocessing**: time is **downsampled** to match the tokenizer's time resolution, tracks of the same programs are merged, notes with pitches outside the tokenizer's pitch range are removed, note velocities and tempos are downsampled, finally notes, tempos and time signatures are deduplicated;
2. **Parsing of global events**: tempos and time signature tokens are created;
3. **Parsing of the tracks events**: notes, chords, controls (pedals...) and tokens specific to each tracks are parsed to create their associated tokens;
4. **Creating time tokens**: the tokens representing the time are created in order to bind the previously created global and track tokens.

The resulting tokens are provided by the tokenizer as one or :class:`miditok.TokSequence` depending on the tokenizer's IO format (:ref:`Tokens & TokSequence input / output format`)

The first three steps are common for all tokenizers, while the fourth is handled independently by each tokenizer.
The first step allows to format the music file so that its content fits the tokenizer's vocabulary before being parsed.


Vocabulary
------------------------

As introduced in :ref:`Tokens and vocabulary`, the vocabulary acts as a lookup table between the tokens (string) and their ids (integers).
It can be accessed with ``tokenizer.vocab`` to get the string to id mapping.

For tokenizers with embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of dictionaries, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

**With a trained tokenizer:**
``tokenizer.vocab`` holds all the basic tokens describing the note and time attributes of music. By analogy with text, this vocabulary can be seen as the alphabet of unique characters.
After :ref:`Training a tokenizer`, a new vocabulary is built with newly created tokens from pairs of basic tokens. This vocabulary can be accessed with ``tokenizer.vocab_model``, and maps tokens as bytes (string) to their associated ids (int). This is the vocabulary of the 🤗tokenizers model.

TokSequence
------------------------

The methods of MidiTok use :class:`miditok.TokSequence` objects as input and outputs. A :class:`miditok.TokSequence` holds tokens as strings, integers, ``miditok.Event`` and bytes (used internally to encode the token ids with trained tokenizers). TokSequences are subscriptable, can be sliced, concatenated and implement the ``__len__`` magic method.

You can use the :py:func:`miditok.MusicTokenizer.complete_sequence` method to automatically fill the non-initialized attributes of a :class:`miditok.TokSequence`.

.. autoclass:: miditok.TokSequence
    :members:


The MusicTokenizer class
------------------------

MidiTok features several MIDI tokenizations, all inheriting from the :class:`miditok.MusicTokenizer` class.
You can customize your tokenizer by creating it with a custom :class:`miditok.TokenizerConfig`.

.. autoclass:: miditok.MusicTokenizer
    :members:


Tokens & TokSequence input / output format
--------------------------------------------

Depending on the tokenizer at use, the **format** of the tokens returned by the :py:func:`miditok.MusicTokenizer.encode` method may vary, as well as the expected format for the :py:func:`miditok.MusicTokenizer.decode` method. The format is given by the :py:func:`miditok.MusicTokenizer.io_format` property. For any tokenizer, the format is the same for both methods.

The format is deduced from the :py:func:`miditok.MusicTokenizer.is_multi_voc` and ``one_token_stream`` tokenizer attributes.
``one_token_stream`` determined wether the tokenizer outputs a unique :class:`miditok.TokSequence` covering all the tracks of a music file or one :class:`miditok.TokSequence` per track. It is equal to ``tokenizer.config.one_token_stream_for_programs``, except for :class:`miditok.MMM` for which it is enabled while ``one_token_stream_for_programs`` is False.
:py:func:`miditok.MusicTokenizer.is_multi_voc` being True means that each "token" within a :class:`miditok.TokSequence` is actually a list of ``C`` "sub-tokens", ``C`` being the number of sub-token classes.

This results in four situations, where ``I`` (instrument) is the number of tracks, ``T`` (token) is the number of tokens and ``C`` (class) the number of subtokens per token step:

* ``is_multi_voc`` and ``one_token_stream`` are both ``False``: ``[I,(T)]``;
* ``is_multi_voc`` is ``False`` and ``one_token_stream`` is ``True``: ``(T)``;
* ``is_multi_voc`` is ``True`` and ``one_token_stream`` is ``False``: ``[I,(T,C)]``;
* ``is_multi_voc`` and ``one_token_stream`` are both ``True``: ``(T,C)``.

**Note that if there is no I dimension in the format, the output of** :py:func:`miditok.MusicTokenizer.encode` **is a** :class:`miditok.TokSequence` **object, otherwise it is a list of** :class:`miditok.TokSequence` **objects (one per token stream / track).**

Some tokenizer examples to illustrate:

* **TSD** without ``config.use_programs`` will not have multiple vocabularies and will treat each track as a unique stream of tokens, hence it will convert music files to a list of :class:`miditok.TokSequence` objects, ``(I,T)`` format.
* **TSD** with ``config.use_programs`` being True will convert all tracks to a single stream of tokens, hence one :class:`miditok.TokSequence` object, ``(T)`` format.
* **CPWord** is a multi-voc tokenizer, without ``config.use_programs`` it will treat each track as a distinct stream of tokens, hence it will convert music files to a list of :class:`miditok.TokSequence` objects with the ``(I,T,C)`` format.
* **Octuple** is a multi-voc tokenizer and converts all track to a single stream of tokens, hence it will convert music files to a :class:`miditok.TokSequence` object, ``(T,C)`` format.


Magic methods
------------------------

`Magic methods <https://rszalski.github.io/magicmethods/>`_ allows to intuitively access to a tokenizer's attributes and methods. We list them here with some examples.

.. autofunction:: miditok.MusicTokenizer.__call__
    :noindex:
..  code-block:: python

    tokens = tokenizer(score)
    score2 = tokenizer(tokens)

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


Save / Load a tokenizer
------------------------

You can save and load a tokenizer, include its configuration and vocabulary. This is especially useful after :ref:`Training a tokenizer`.

.. autofunction:: miditok.MusicTokenizer.save
    :noindex:

To load a tokenizer from saved parameters, just use the ``params`` argument when creating a it:

..  code-block:: python

    tokenizer = REMI(params=Path("to", "tokenizer.json"))
