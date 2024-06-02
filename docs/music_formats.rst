===================================
Music formats
===================================

This page introduces the basic concepts of music, the MIDI protocol and sequential deep learning models. It aims to bring the basic knowledge around this subjects in order to understand how to use music with AI models, without going into too specific details, for which more comprehensive references are attached.

Music: symbolic and audio
---------------------------

Music is a unique modality in the way that it can take two different forms: symbolic and audio.

Symbolic music represents the successions of notes, arranged in time and along with other musical elements such as tempos and time signatures typically found in the western music notations. The `sheet music <https://en.wikipedia.org/wiki/Sheet_music>`_ is the historical handwritten or printed representation of music that shows the notes on staves from left to right and up and down, with the time and key signatures indicated at the beginning.

.. image:: /assets/bases/sheet_music.png
  :width: 800
  :alt: A sheet music.

The `pianoroll <https://en.wikipedia.org/wiki/Piano_roll>`_ is another symbolic representation which consists of a two axis grid with one axis for the time and one for the note pitches. It was originally used in player pianos, and is now used in most `Digital Audio Wordstation (DAW) <https://en.wikipedia.org/wiki/Digital_audio_workstation>`_ softwares to show the notes and other effects of a track.

.. image:: /assets/bases/pianoroll_daw.png
  :width: 800
  :alt: A piano roll view in the Logic Pro X DAW.

Audio on the other hand represents the *physical* form of music, i.e. a sound signal, more specifically vibrations propagating in a material. Audio music is usually represented as waveforms (time domain) or spectrograms (frequency domain).

A waveform is stricticly the amplitude of a sound as a function of time. In the real world, a waveform is purely continuous. A digital audio waveform as found in audio files such as mp3s will feature a sampling frequency which indicates the number of samples per second used to represent this waveform. This time resolution is usually at least 44.1k samples per seconds, following the `Nyquist–Shannon theorem <https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem>`_ .

A sound, wether from an instrument, a human voice or a music arrangement, is a superposition of many periodic frequencies, defined by their wavelength, amplitude and phase. A spectrogram depicts the intensity in dB of the frequencies as a function of time. It allow to have a representation of these frequencies which is useful when analyzing sound. It can be computed with a `Fourier Transform <https://en.wikipedia.org/wiki/Fourier_transform>`_ , usually a `Short Time Fourier Transform (STFT) <https://ieeexplore.ieee.org/document/1164317>`_ .

.. image:: /assets/bases/spectrogram.png
  :width: 800
  :alt: The spectrogram of a sound, abscissa is time, ordinate is frequency and the color represents the intensity in dB.

Symbolic music can be seen as both discrete and continuous as it represent discrete notes that feature however "continuous-like" attributes, and potentially with a high time resolution (in samples per beat or other specific time duration). For this reason, it is more commonly used with discrete sequential models (e.g. `Transformers <https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`_ ) by being represented as sequences of tokens, which is the purpose of MidiTok. Pianoroll has also been used with `Convolutional Neural Networks (CNNs) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ in past works (e.g. `MuseGan <https://aaai.org/papers/11312-musegan-multi-track-sequential-generative-adversarial-networks-for-symbolic-music-generation-and-accompaniment/>`_ ) but is now uncommon due to the limitations it imposes on the representation of musical elements.

On the other hand, audio is by nature a continuous modality, as it represent the waveform of the sound itself. From a practical point of view, modeling raw waveforms with neural networks is often intractable due to the high time resolution of audio, despite works that achieved to do it (`WaveNet <https://arxiv.org/pdf/1609.03499>`_ , `Jukebox <https://openai.com/index/jukebox/>`_ ). For this reason, audio has been more commonly formatted as spectrograms when used with neural networks, and used with CNNs as it conventiently takes the form of a 2-dimensional matrix with distinct continuous patterns like images.
Research in neural audio codecs allowed to "compress" audio waveform into a reduced number of discrete values allows to use waveforms as sequences of tokens with discrete models such as Transformers. For more details, see `SoundStream <https://ieeexplore.ieee.org/document/9625818>`_ and `EnCodec <https://openreview.net/forum?id=ivCd8z8zR2>`_ which are respectively used with `MusicLM <https://arxiv.org/abs/2301.11325>`_ and `MusicGen <https://proceedings.neurips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html>`_ .


Symbolic music files format
-----------------------------

MIDI :ref:`midi-protocol-label`_
ABC
MusicXML
