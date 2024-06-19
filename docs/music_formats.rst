===================================
Music formats
===================================

This page introduces the two representations of music and symbolic music file formats. It aims to present the basic differences between audio and symbolic music in order to better understand how they can be used with AI models, without going to much in the details, for which more comprehensive references are attached.

Music: symbolic and audio
---------------------------

Music is a unique modality in the way that it can take two different forms: symbolic and audio.

Symbolic music represents the successions of notes, arranged in time and along with other musical elements such as tempos and time signatures typically found in the western music notations. The `sheet music <https://en.wikipedia.org/wiki/Sheet_music>`_ is the historical handwritten or printed representation of music that shows the notes on staves from left to right and up and down, with the time and key signatures indicated at the beginning.

.. image:: /assets/bases/sheet_music.png
  :width: 800
  :alt: A sheet music.

The `pianoroll <https://en.wikipedia.org/wiki/Piano_roll>`_ is another symbolic representation which consists of a two axis grid with one axis for the time and one for the note pitches. It was originally used in player pianos, and is now used in most `Digital Audio Wordstation (DAW) <https://en.wikipedia.org/wiki/Digital_audio_workstation>`_ software to show the notes and other effects of a track.

.. image:: /assets/bases/pianoroll_daw.png
  :width: 800
  :alt: A piano roll view in the Logic Pro X DAW.

Audio on the other hand represents the *physical* form of music, i.e. a sound signal, more specifically vibrations propagating in a material. Audio music is usually represented as waveforms (time domain) or spectrograms (frequency domain).

A waveform is strictly the amplitude of a sound as a function of time. In the real world, a waveform is purely continuous. A digital audio waveform as found in audio files such as mp3s will feature a sampling frequency which indicates the number of samples per second used to represent this waveform. This time resolution is usually at least 44.1k samples per seconds, following the `Nyquist–Shannon theorem <https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem>`_ .

A sound, whether from an instrument, a human voice or a music arrangement, is a superposition of many periodic frequencies, defined by their wavelength, amplitude and phase. A spectrogram depicts the intensity in dB of the frequencies as a function of time. It allow to have a representation of these frequencies which is useful when analyzing sound. It can be computed with a `Fourier Transform <https://en.wikipedia.org/wiki/Fourier_transform>`_ , usually a `Short Time Fourier Transform (STFT) <https://ieeexplore.ieee.org/document/1164317>`_ .

.. image:: /assets/bases/spectrogram.png
  :width: 800
  :alt: The spectrogram of a sound, abscissa is time, ordinate is frequency and the color represents the intensity in dB.

Symbolic music can be seen as both discrete and continuous as it represent discrete notes that feature however "continuous-like" attributes, and potentially with a high time resolution (in samples per beat or other specific time duration). **For this reason, it is more commonly used with discrete sequential models**, which we introduce in :ref:`sequential-models-label`), **by being represented as sequences of tokens**, which is the purpose of MidiTok. Pianoroll has also been used with `Convolutional Neural Networks (CNNs) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ in past works (e.g. `MuseGan <https://aaai.org/papers/11312-musegan-multi-track-sequential-generative-adversarial-networks-for-symbolic-music-generation-and-accompaniment/>`_ ) but is now uncommon due to the limitations it imposes on the representation of musical elements.

On the other hand, audio is by nature a continuous modality, as it represent the waveform of the sound itself. From a practical point of view, modeling raw waveforms with neural networks is often intractable due to the high time resolution of audio, despite works that achieved to do it (`WaveNet <https://arxiv.org/pdf/1609.03499>`_ , `Jukebox <https://openai.com/index/jukebox/>`_ ). For this reason, audio has been more commonly formatted as spectrograms when used with neural networks, and used with CNNs as it conveniently takes the form of a 2-dimensional matrix with distinct continuous patterns like images.
Research in neural audio codecs allowed to "compress" audio waveform into a reduced number of discrete values allows to use waveforms as sequences of tokens with discrete models such as Transformers. For more details, see `SoundStream <https://ieeexplore.ieee.org/document/9625818>`_ and `EnCodec <https://openreview.net/forum?id=ivCd8z8zR2>`_ which are respectively used with `MusicLM <https://arxiv.org/abs/2301.11325>`_ and `MusicGen <https://proceedings.neurips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html>`_ .


Symbolic music files format
-----------------------------

There are three major file formats for symbolic music: MIDI, abc and musicXML. **MidiTok supports MIDI and abc files.**

MIDI, standing for *Musical Instrument Digital Interface*, is a digital communication protocol standard in the music sector. It describes the protocol itself, the physical connector to transmit the protocol between devices, and a digital file format.
A MIDI file allows to store MIDI messages as a symbolic music file. It is the most abundant file format among available music datasets. It is the most comprehensive and versatile file format for musical music, as such we present it more in detail in :ref:`midi-protocol-label`.


The ABC notation is a notation for symbolic music, and a file format with the extension ``abc``. Its simplicity has made it widely used to write and share traditional and folk tunes from Western Europe.
Each tune begins with a few lines indicating its title, time signature, default note length, key and others. Lines following the key represent the notes. A note is indicated by its letter, followed by a ``/x`` or ``x`` to respectively divide or multiply its length by ``x`` :math:`\in \mathbb{N}^{\star}` compared to the default note length. An upper case (e.g., A) means a pitch one octave below than a lower case (a).

MusicXML is an open file format and music notation. Inspired by the XML file format, it is structured with the same item-hierarchy. An example is shown below.

..  code-block:: xml

    <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <!DOCTYPE score-partwise PUBLIC
        "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
        "http://www.musicxml.org/dtds/partwise.dtd">
    <score-partwise version="3.1">
        <part-list>
            <score-part id="P1">
                <part-name>Music</part-name>
            </score-part>
        </part-list>
        <part id="P1">
            <measure number="1">
                <attributes>
                    <divisions>1</divisions>
                    <key><fifths>0</fifths></key>
                    <time><beats>4</beats><beat-type>4</beat-type></time>
                    <clef><sign>G</sign><line>2</line></clef>
                </attributes>
                <note>
                    <pitch><step>C</step><octave>4</octave></pitch>
                    <duration>4</duration>
                    <type>whole</type>
                </note>
            </measure>
        </part>
    </score-partwise>

The ``part-list`` references the parts to be written following with the tag ``part``. A ``measure`` is defined with its attributes, followed by notes and their attributes.
The common file extensions are ``.mxl`` and ``.musicxml``.
