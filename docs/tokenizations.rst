=================
Tokenizations
=================

This page details the tokenizations featured by MidiTok. They inherit from :ref:`MIDITokenizer`, see the documentation for learn to use the common methods. For each of them, the token equivalent of the lead sheet below is showed.

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/music_sheet.png?raw=true
  :width: 800
  :alt: Music sheet example

REMI
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/remi.png?raw=true
  :width: 800
  :alt: REMI sequence, time is tracked with Bar and position tokens

.. autoclass:: miditok.REMI
    :noindex:
    :show-inheritance:

MIDI-Like
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/midi_like.png?raw=true
  :width: 800
  :alt: MIDI-Like token sequence, with TimeShift and NoteOff tokens

.. autoclass:: miditok.MIDILike
    :noindex:
    :show-inheritance:

TSD
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/tsd.png?raw=true
  :width: 800
  :alt: TSD sequence, like MIDI-Like with Duration tokens

.. autoclass:: miditok.TSD
    :noindex:
    :show-inheritance:

Structured
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/structured.png?raw=true
  :width: 800
  :alt: Structured tokenization, the token types always follow the same succession pattern

.. autoclass:: miditok.Structured
    :noindex:
    :show-inheritance:

Compound Word (CPWord)
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/cp_word.png?raw=true
  :width: 800
  :alt: CP Word sequence, tokens of the same family are grouped together

.. autoclass:: miditok.CPWord
    :noindex:
    :show-inheritance:

Octuple
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/octuple.png?raw=true
  :width: 800
  :alt: Octuple sequence, with a bar and position embeddings

.. autoclass:: miditok.Octuple
    :noindex:
    :show-inheritance:

Octuple Mono
------------------------

.. autoclass:: miditok.OctupleMono
    :noindex:
    :show-inheritance:

MuMIDI
------------------------

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/mumidi.png?raw=true
  :width: 800
  :alt: MuMIDI sequence, with a bar and position embeddings

.. autoclass:: miditok.MuMIDI
    :noindex:
    :show-inheritance:
