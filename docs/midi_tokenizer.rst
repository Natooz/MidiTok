=================
MIDI Tokenizer
=================

Base tokenizer class
------------------------

MidiTok features several MIDI tokenizations, that are all built from a common ``MIDITokenizer`` class.

.. autofunction:: miditok.MIDITokenizer.token_types_errors

Otherwise, :py:func:`miditok.MIDITokenizer.token_types_errors` ref

For example:

>>> from miditok import REMI
>>> from miditok.utils import get_midi_programs
>>> from miditoolkit import MidiFile
>>>
>>> # Our parameters
>>> pitch_range = range(21, 109)
>>> beat_res = {(0, 4): 8, (4, 12): 4}
>>> nb_velocities = 32
>>> additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
>>>                      'rest_range': (2, 8),  # (half, 8 beats)
>>>                      'nb_tempos': 32,  # nb of tempo bins
>>>                      'tempo_range': (40, 250)}  # (min, max)
>>>
>>> # Creates the tokenizer and loads a MIDI
>>> tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
>>> midi = MidiFile('path/to/your_midi.mid')
>>>
>>> # Converts MIDI to tokens, and back to a MIDI
>>> tokens = tokenizer(midi)  # automatically detects MIDIs and tokens before converting
>>> converted_back_midi = tokenizer(tokens, get_midi_programs(midi))
>>>
>>> # Converts just a selected track
>>> tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
>>> piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])
>>>
>>> # And convert it back (the last arg stands for (program number, is drum))
>>> converted_back_track, tempo_changes = tokenizer.tokens_to_track(piano_tokens, midi.ticks_per_beat, (0, False))
>>> lumache.get_random_ingredients()
>>> ['shells', 'gorgonzola', 'parsley']

Vocabulary
------------------------

Vocab

Tokenize dataset
------------------------

Tokenize dataset

Magic methods
------------------------

call, len, get...

Save / Load tokenizer
------------------------

Save / load
