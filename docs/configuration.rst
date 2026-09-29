=======================
Tokenizer Configuration
=======================

MidiTok's tokenizers can be customized with a wide variety of options, and most of the preprocessing and downsampling steps can be tailored to your specifications.

Tokenizer config
------------------------

All tokenizers are initialized with common parameters, that are hold in a :class:`miditok.TokenizerConfig` object, documented below. A tokenizer's configuration can be accessed with ``tokenizer.config``.
Some tokenizers might take additional specific arguments / parameters when creating them.

.. autoclass:: miditok.TokenizerConfig
    :members:


How MidiTok handles time
----------------------------

MidiTok handles time by resampling the music file's time division (time resolution) to a new resolution determined by the ``beat_res`` attribute of of :class:`miditok.TokenizerConfig`. This argument determines which time tokens are present in the vocabulary.

It allows to create ``Duration`` and ``TimeShift`` tokens with different resolution depending on their values. It is typically common to use higher resolutions for short time duration (i.e. short values will be represented with greater accuracy) and lower resolutions for higher time values (that generally do not need to be represented with great accuracy).
The values of these tokens take the form of tuple as: ``(num_beats, num_samples, resolution)``. For instance, the time value of token ``(2, 3, 8)`` corresponds to 2 beats and 3/8 of a beat. ``(2, 2, 4)`` corresponds to 2 beats and half of a beat (2.5).

For position-based tokenizers, the number of ``Position`` in the vocabulary is equal to the maximum resolution found in ``beat_res``.

An example of the downsampling applied by MidiTok during the preprocessing is shown below.

..  figure:: /assets/midi_preprocessing_original.png
    :alt: Original MIDI file
    :width: 800

    Original MIDI file from the `Maestro dataset <https://magenta.tensorflow.org/datasets/maestro>`_ with a 4/4 time signature. The numbers at the top indicate the bar number (125) followed by the beat number within the bar.

..  figure:: /assets/midi_preprocessing_preprocessed.png
    :alt: Downsampled MIDI file.
    :width: 800

    MIDI file with time downsampled to 8 samples per beat.

Additional tokens
------------------------

MidiTok offers to include additional tokens on music information. You can specify them in the ``tokenizer_config`` argument (:class:`miditok.TokenizerConfig`) when creating a tokenizer. The :class:`miditok.TokenizerConfig` documentations specifically details the role of each of them, and their associated parameters.

.. csv-table:: Compatibility table of tokenizations and additional tokens.
   :file: additional_tokens_table.csv
   :header-rows: 1

¹: using both time signatures and rests with :class:`miditok.CPWord` might result in time alterations, as the time signature changes are carried with the Bar tokens which can be skipped during period of rests.
²: using time signatures with :class:`miditok.Octuple` might result in time alterations, as the time signature changes are carried with the note onsets. An example is shown below.

The ``use_key_signatures`` option enables ``KeySig`` tokens representing the key signature of the music, formatted as ``KeySig_{key}:{tonality}`` where ``key`` is the number of accidentals (from ``-7`` to ``7``) and ``tonality`` is ``0`` for major or ``1`` for minor. Key signatures are global events, stored on ``symusic.Score.key_signatures``. A default C major key signature is added at tick 0 if the ``Score`` does not contain any. This option is supported by :class:`miditok.REMI`, :class:`miditok.TSD`, :class:`miditok.MIDILike` and :class:`miditok.PerTok`; it is disabled with a warning for the other tokenizations. It follows the Standard MIDI File key signature meta event (`MIDI key signature meta message <https://www.recordingblogs.com/wiki/midi-key-signature-meta-message>`_).

Alternatively, **Velocity** and **Duration** tokens are optional and are enabled by default for all tokenizers.

.. image:: /assets/Octuple_TS_Rest/original.png
  :width: 800
  :alt: Original MIDI sample preprocessed / downsampled

.. image:: /assets/Octuple_TS_Rest/tokenized.png
  :width: 800
  :alt: MIDI sample after being tokenized, the time has been shifted to a bar during the time signature change

Below is an example of how pitch intervals would be tokenized, with a ``max_pitch_interval`` of 15.

.. image:: /assets/pitch_intervals.png
  :width: 800
  :alt: Schema of the pitch intervals over a piano-roll

**Control changes** can be tokenized with the ``use_control_changes`` option. Each message becomes a ``ControlChange_{number}-{value}`` token. By default, the control numbers listed in ``miditok.constants.CONTROL_CHANGES`` are selected; use ``control_change_numbers`` to choose a subset or include other numbers. Control changes are supported by :ref:`MIDI-Like`, :ref:`REMI`, :ref:`TSD` and :class:`miditok.PerTok`.

Values are handled in three categories. These are MidiTok preprocessing policies for 7-bit MIDI 1.0 CC messages, rather than universal MIDI controller types; this option does not implement native 32-bit MIDI 2.0 CC messages.

* **Continuous:** the numbers in ``CONTINUOUS_CONTROL_CHANGE_NUMBERS`` are quantized to ``control_change_n_bins`` evenly spaced integer levels, including 0 and 127. The default is 32 bins; any integer from 2 to 128 is supported. Set 128 to preserve all continuous values. CC64 (sustain) and CC67 (soft/una corda) retain intermediate pedal values, following the `MIDI-CI Piano Profile v1.0 <https://midi.org/midi-ci-profile-for-piano>`_ (sections 4.2 and 7.1.2).
* **Switch:** the numbers in ``SWITCH_CONTROL_CHANGE_NUMBERS`` (65, 66, 68, 69 and 122) use only 0 and 127. Input values below 64 become 0; values of 64 or higher become 127. CC122 formally defines only 0 and 127; applying the threshold to its other values is MidiTok's normalization policy.
* **Discrete:** all other numbers retain the complete 0–127 value range, including selectors, commands, data-entry messages, fine-resolution bytes and unknown controllers. This preserves their raw values; it does not mean every value is defined by the MIDI specification for every command.

Preprocessing preserves repeated messages and the input order of CCs at equal times. This matters for increment/decrement commands and parameter-selection sequences. See the `MIDI control change definitions <https://midi.org/midi-1-0-control-change-messages>`_.

When CC64 is selected and control-change tokenization is enabled, it takes precedence over the legacy pedal representation. If ``use_sustain_pedals`` is also enabled, the tokenizer emits a warning and disables it along with ``sustain_pedal_duration`` before building the vocabulary. When CC64 is excluded or control changes are disabled, the existing pedal arguments and MIDI export behavior remain available.


Special tokens
------------------------

MidiTok offers to include some special tokens to the vocabulary. These tokens with no "musical" information can be used for training purposes.
To use special tokens, you must specify them with the ``special_tokens`` argument when creating a tokenizer. By default, this argument is set to ``["PAD", "BOS", "EOS", "MASK"]``. Their signification are:

* **PAD** (``PAD_None``): a padding token to use when training a model with batches of sequences of unequal lengths. The padding token id is often set to 0. If you use Hugging Face models, be sure to pad inputs with this tokens, and pad labels with *-100*.
* **BOS** (``SOS_None``): "Start Of Sequence" token, indicating that a token sequence is beginning.
* **EOS** (``EOS_None``): "End Of Sequence" tokens, indicating that a token sequence is ending. For autoregressive generation, this token can be used to stop it.
* **MASK** (``MASK_None``): a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like `BERT <https://arxiv.org/abs/1810.04805>`_.

**Note:** you can use the ``tokenizer.special_tokens`` property to get the list of the special tokens of a tokenizer, and ``tokenizer.special_tokens`` for their ids.
