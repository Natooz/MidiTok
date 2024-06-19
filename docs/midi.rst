.. _midi-protocol-label:

===================================
The MIDI protocol
===================================

MIDI, standing for *Musical Instrument Digital Interface*, is a digital communication protocol standard in the music sector. It describes the protocol itself, the physical connector to transmit the protocol between devices, and a digital file format.
A MIDI file allows to store MIDI messages as a symbolic music file. It is the most abundant file format among available music datasets.

History of MIDI
-----------------------------

MIDI first appeared in the early eighties, when digital instrument manufacturers needed a digital protocol for communication between devices such as synthesizers and computers. It was standardized in 1983 by the first specifications, and is currently maintained by the `MIDI Manufacturers Association <https://www.midi.org>`_\. Meanwhile `new specifications <https://www.midi.org/specifications>`_ were made, the two major ones and still the norm today being the General MIDI 1 (GM1) and General MIDI 2 (GM2). These specifications aim to guide the manufacturers to design digital music devices compatible with the ones from other manufacturers by making sure they implement the protocol by following the same recommendations.

The MIDI protocol allows to represent **notes, tempos, time signatures, key signatures, instruments (called programs) and effects (called controls) such as sustain pedal, pitch bend or modulation.**
MIDI is an event based protocol. It consists of a series of messages, which can occur in multiple channels. Each message is composed of two key information, 1) the delta time expressed, which is the distance in ticks with the previous event (in the same channel) and so represents its position in time, 2) a series of bytes which represents its content.

The latest evolution of the MIDI protocol is the MIDI Polyphonic Expression (shortly called MPE). This new norm allows manufacturers to create MIDI devices on which a specific channel is assigned to each note allowing the user to apply pitch bend and modulation on each key independently. These devices are typically built with touch-sensitive keys. The MIDI Manufacturers Association released the complete `specifications <https://www.midi.org/midi-articles/midi-polyphonic-expression-mpe>`_ on March 2018.


MIDI Messages
-----------------------------

A message expresses an event or an information. It takes the form of a series of bytes. The first is the Status byte which specifies the type of message and the channel, followed by one or two data bytes which contain the information. All the messages and their significations are described in the GM1 and GM2 specifications. The most important are:

- *Note On*: a note is being played, specifies its pitch and velocity;
- *Note Off*: a note is released, specifies the note (by its pitch) to stop and the velocity;
- *Time Signature Change*: indicates the current time signature;
- *Tempo Change*: indicates the current tempo;
- *Program Change*: specifies the current instrument being played;
- *Control Change*: a control parameter is modified or applied. The modulation wheel, foot sustain pedal, volume control or bank select are for instance effects transcribed into Control Change messages.

Note that these messages are "voice messages", which means that each of them is applied within a channel that is specified in its status byte. The MIDI protocol handles up to sixteen channels which allows to connect multiple devices that are playing and communicating simultaneously. The channel 10 is reserved for drums, which is a specific "program" in which the pitch values corresponds to drum sounds like kicks, snares, or hi-hats.

Time in MIDI
-----------------------------

Time in MIDI is determined by its **time division**, which is a clock signal expressed in **ticks per quarter note** (tpq), and can be seen as a time resolution. Common time division values are 384, 480 and 960 tpq as they are divisible by 3, 4, 6 and 8 which are common time signature numerators and denominators.
The time division can also be set in ticks per second, but this option is more rarely encountered as it makes less sense to use seconds as the tempo and time signature are known in MIDI.
The time division is the first information that can be read at the beginning of a file, and a MIDI file can only have one time division.

The number of ticks per bar and ticks per beat can be calculated from the MIDI's time division (:math:`time_{div}`) and the current time signature (:math:`\frac{ts_{num}}{ts_{denom}}`):

- :math:`tpbeat = time_{div} \times \frac{4}{ts_{denom}}`
- :math:`tpbar = tpbeat \times ts_{num}`

Hence, for a :math:`\frac{4}{4}` time signature, the number of ticks per beat is equal to the time division (as a beat is equal to a quarter note) and the number of ticks per bar is equal to four times the number of ticks per beat.
