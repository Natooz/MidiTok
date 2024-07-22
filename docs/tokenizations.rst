=================
Tokenizations
=================

This page details the tokenizations featured by MidiTok. They inherit from :class:`miditok.MusicTokenizer`, see the documentation for learn to use the common methods. For each of them, the token equivalent of the lead sheet below is showed.

.. image:: /assets/music_sheet.png
  :width: 800
  :alt: Music sheet example

REMI
------------------------

.. image:: /assets/remi.png
  :width: 800
  :alt: REMI sequence, time is tracked with Bar and position tokens

.. autoclass:: miditok.REMI
    :show-inheritance:

REMIPlus
------------------------

REMI+ is an extended version of :ref:`REMI` (Huang and Yang) for general multi-track, multi-signature symbolic music sequences, introduced in `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`_, which handles multiple instruments by adding ``Program`` tokens before the ``Pitch`` ones.

You can get the REMI+ tokenization by using the :ref:`REMI` tokenizer with ``config.use_programs``, ``config.one_token_stream_for_programs`` and ``config.use_time_signatures`` enabled.

MIDI-Like
------------------------

.. image:: /assets/midi_like.png
  :width: 800
  :alt: MIDI-Like token sequence, with TimeShift and NoteOff tokens

.. autoclass:: miditok.MIDILike
    :show-inheritance:

TSD
------------------------

.. image:: /assets/tsd.png
  :width: 800
  :alt: TSD sequence, like MIDI-Like with Duration tokens

.. autoclass:: miditok.TSD
    :show-inheritance:

Structured
------------------------

.. image:: /assets/structured.png
  :width: 800
  :alt: Structured tokenization, the token types always follow the same succession pattern

.. autoclass:: miditok.Structured
    :show-inheritance:

CPWord
------------------------

.. image:: /assets/cp_word.png
  :width: 800
  :alt: CP Word sequence, tokens of the same family are grouped together

.. autoclass:: miditok.CPWord
    :show-inheritance:

Octuple
------------------------

.. image:: /assets/octuple.png
  :width: 800
  :alt: Octuple sequence, with a bar and position embeddings

.. autoclass:: miditok.Octuple
    :show-inheritance:

MuMIDI
------------------------

.. image:: /assets/mumidi.png
  :width: 800
  :alt: MuMIDI sequence, with a bar and position embeddings

.. autoclass:: miditok.MuMIDI
    :show-inheritance:

MMM
------------------------

.. autoclass:: miditok.MMM
    :show-inheritance:

PerTok
------------------------

.. autoclass:: miditok.PerTok
    :show-inheritance:


Create yours
------------------------

You can easily create your own tokenizer and benefit from the MidiTok framework. Just create a class inheriting from :class:`miditok.MusicTokenizer`, and override:

* :py:func:`miditok.MusicTokenizer._add_time_events` to create time events from global and track events;
* :py:func:`miditok.MusicTokenizer._tokens_to_score` to decode tokens into a ``Score`` object;
* :py:func:`miditok.MusicTokenizer._create_vocabulary` to create the tokenizer's vocabulary;
* :py:func:`miditok.MusicTokenizer._create_token_types_graph` to create the possible token types successions (used for eval only).

If needed, you can override the methods:

* :py:func:`miditok.MusicTokenizer._score_to_tokens` the main method calling specific tokenization methods;
* :py:func:`miditok.MusicTokenizer._create_track_events` to include special track events;
* :py:func:`miditok.MusicTokenizer._create_global_events` to include special global events.

If you think people can benefit from it, feel free to send a pull request on `Github <https://github.com/Natooz/MidiTok>`_.
