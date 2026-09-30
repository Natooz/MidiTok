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

BEAT
------------------------

BEAT (Beat-wise Encoding for Autoregressive Transformers) describes music one beat at a time. It was introduced in `BEAT (Qian et al.) <https://openreview.net/forum?id=XrrGXLksji>`_.

A **Pattern describes what one pitch does during one beat**, split into four equal steps. MidiTok defines this beat using the time-signature denominator: a quarter note in 4/4, an eighth note in 6/8, or a half note in 2/2. At each step, record:

* **0:** silent.
* **1:** a note starts.
* **2:** the note continues.

For example, a note starts, lasts three steps, then stops:

.. code-block:: text

    Step:     1      2      3      4
    Action:  Start  Hold   Hold   Silence
    State:    1      2      2      0

These four digits are packed into one number using weights 27, 9, 3 and 1:

.. code-block:: text

    1 x 27 + 2 x 9 + 2 x 3 + 0 x 1 = 51

This becomes **Pattern_51**. The preceding ``Pitch`` token identifies which pitch it describes. This packing is just a base-3 number, because each step has three possible states.

**Why four steps?** This is the chosen timing resolution. It keeps the pattern vocabulary small:

.. list-table:: Timing resolution and pattern vocabulary
    :header-rows: 1

    * - Steps per beat
      - Possible patterns
    * - 4
      - 3⁴ = 81
    * - 8
      - 3⁸ = 6,561

More steps give finer timing but rapidly increase the vocabulary. MidiTok's BEAT fixes the resolution to four steps, overriding ``beat_res``, and uses pattern values from 0 to 80.

BEAT always enables time signatures, enforcing ``use_time_signatures=True`` even if the configuration sets it to ``False``. Each complete bar contains as many beat groups as the time-signature numerator. The step length follows the denominator:

.. list-table:: Beat groups and steps by time signature
    :header-rows: 1

    * - Time signature
      - Groups per complete bar
      - One group
      - One step
    * - 4/4
      - 4
      - Quarter note
      - Sixteenth note
    * - 6/8
      - 6
      - Eighth note
      - Thirty-second note
    * - 3/8
      - 3
      - Eighth note
      - Thirty-second note
    * - 2/2
      - 2
      - Half note
      - Eighth note

This follows MidiTok's denominator-based convention rather than the perceived musical pulse: 6/8 has six eighth-note groups here, although it is usually felt as two dotted-quarter beats. It also differs from the authors' reference implementation, which fixes each group to a quarter note. Time-signature changes update the group and step lengths at bar boundaries. Missing time signatures default to 4/4.

This four-step resolution **does not limit note length**: a note can continue into following beats. For example, ``2222`` (``Pattern_80``) means "held throughout this beat." Notes spanning beats use these continuation states instead of ``Duration`` or ``NoteOff`` tokens.

The sequence uses ``Pitch``, ``Pattern`` and optional ``Velocity`` triples, with explicit ``Bar`` and ``Beat`` markers and ``Rest_None`` for empty beats. Within each track and beat, pitches descend: the first pitch is absolute and subsequent pitch values are downward intervals. Tracks are grouped inside each beat in program order, with drums last, following Section 3.1 and the authors' reference implementation. MidiTok's ``Program`` tokens prefix each track's content within every beat, independently of the ``program_changes`` option.

Velocity is averaged over active subdivisions for each pitch and beat, then quantized to the configured velocity bins. Decoding uses the onset beat's velocity. Overlapping notes of the same pitch are truncated at the next onset. Time signatures and optional key signatures are supported. Tempo, time-signature and key-signature changes are delayed to bar boundaries; key changes mapping to the same bar keep the last value.

Duration, rest and relative-pitch encoding are intrinsic to BEAT; the optional flags for these features do not change its representation. Pedals, control changes, pitch bends, chords and attribute controls are not supported.

BEAT always uses one token stream with program tokens, enforcing ``use_programs=True`` and ``one_token_stream_for_programs=True``. Tracks sharing a program (including multiple drum tracks) are preserved separately instead of being merged during preprocessing. As a MidiTok extension, their input order identifies them across beats: each emits a ``Program`` block in every beat, with ``Rest_None`` when silent. These silent blocks keep subsequent notes and sustains attached to the correct track without adding track-ID tokens. Tracks with unique programs still omit silent blocks.

.. autoclass:: miditok.BEAT
    :show-inheritance:

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
