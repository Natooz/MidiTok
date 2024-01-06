=================
Tokenizations
=================

This page details the tokenizations featured by MidiTok. They inherit from :class:`miditok.MIDITokenizer`, see the documentation for learn to use the common methods. For each of them, the token equivalent of the lead sheet below is showed.

.. image:: /assets/music_sheet.png
  :width: 800
  :alt: Music sheet example

REMI
------------------------

.. image:: /assets/remi.png
  :width: 800
  :alt: REMI sequence, time is tracked with Bar and position tokens

.. autoclass:: miditok.REMI
    :noindex:
    :show-inheritance:

REMIPlus
------------------------

REMI+ is an extended version of :ref:`REMI` (Huang and Yang) for general multi-track, multi-signature symbolic music sequences, introduced in `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which handle multiple instruments by adding ``Program`` tokens before the ``Pitch`` ones.

In the previous versions of MidiTok, we used to implement REMI+ as a dedicated class. Now that all the tokenizers supports the additional tokens in a more flexible way, you can get the REMI+ tokenization by using the :ref:`REMI` tokenizer with ``config.use_programs`` and ``config.one_token_stream_for_programs`` and ``config.use_time_signatures`` set to ``True``.

MIDI-Like
------------------------

.. image:: /assets/midi_like.png
  :width: 800
  :alt: MIDI-Like token sequence, with TimeShift and NoteOff tokens

.. autoclass:: miditok.MIDILike
    :noindex:
    :show-inheritance:

TSD
------------------------

.. image:: /assets/tsd.png
  :width: 800
  :alt: TSD sequence, like MIDI-Like with Duration tokens

.. autoclass:: miditok.TSD
    :noindex:
    :show-inheritance:

Structured
------------------------

.. image:: /assets/structured.png
  :width: 800
  :alt: Structured tokenization, the token types always follow the same succession pattern

.. autoclass:: miditok.Structured
    :noindex:
    :show-inheritance:

CPWord
------------------------

.. image:: /assets/cp_word.png
  :width: 800
  :alt: CP Word sequence, tokens of the same family are grouped together

.. autoclass:: miditok.CPWord
    :noindex:
    :show-inheritance:

Octuple
------------------------

.. image:: /assets/octuple.png
  :width: 800
  :alt: Octuple sequence, with a bar and position embeddings

.. autoclass:: miditok.Octuple
    :noindex:
    :show-inheritance:

MuMIDI
------------------------

.. image:: /assets/mumidi.png
  :width: 800
  :alt: MuMIDI sequence, with a bar and position embeddings

.. autoclass:: miditok.MuMIDI
    :noindex:
    :show-inheritance:

MMM
------------------------

.. autoclass:: miditok.MMM
    :noindex:
    :show-inheritance:


Create yours
------------------------

You can easily create your own tokenizer and benefit from the MidiTok framework. Just create a class inheriting from :class:`miditok.MIDITokenizer`, and override the :py:func:`miditok.MIDITokenizer._add_time_events`, :py:func:`miditok.MIDITokenizer._tokens_to_midi`, :py:func:`miditok.MIDITokenizer._create_vocabulary` and :py:func:`miditok.MIDITokenizer._create_token_types_graph` (and optionally if needed :py:func:`miditok.MIDITokenizer._midi_to_tokens`, :py:func:`miditok.MIDITokenizer._create_track_events` and :py:func:`miditok.MIDITokenizer._create_midi_events`) methods with your tokenization strategy.

If you think people can benefit from it, feel free to send a pull request on `Github <https://github.com/Natooz/MidiTok>`_.
