""" Structured MIDI encoding method as using in the Piano Inpainting Application
https://arxiv.org/abs/2107.05944

"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Event
from .constants import *


class StructuredEncoding(MIDITokenizer):
    """ Structured MIDI encoding method as using in the Piano Inpainting Application
    https://arxiv.org/abs/2107.05944
    The token types follows the specific pattern:
    Pitch -> Velocity -> Duration -> Time Shift -> back to Pitch ...
    NOTE: this encoding uses only "Time Shifts" events to move in the time, and only
    from one note to another. Hence it is suitable to encode continuous sequences of
    notes without long periods of silence. If your dataset contains music with long
    pauses, you might handle them with an appropriate "time shift" dictionary
    (which values are made from the beat_res dict) or with a different encoding.

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param program_tokens: will add entries for MIDI programs in the dictionary, to use
            in the case of multitrack generation for instance
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, program_tokens: bool = PROGRAM_TOKENS, params=None):
        # No additional tokens
        additional_tokens = {'Chord': False, 'Rest': False, 'Tempo': False}
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def track_to_tokens(self, track: Instrument) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        events = []

        dur_bins = self.durations_ticks[self.current_midi_metadata['time_division']]

        # First time shift if needed
        if track.notes[0].start != 0:
            if track.notes[0].start > max(dur_bins):
                time_shift = track.notes[0].start % self.current_midi_metadata['time_division']  # beat wise
            else:
                time_shift = track.notes[0].start
            index = np.argmin(np.abs([ticks - time_shift for ticks in dur_bins]))
            events.append(Event(
                name='Time-Shift',
                time=0,
                value='.'.join(map(str, self.durations[index])),
                text=f'{time_shift} ticks'))

        # Creates the Pitch, Velocity, Duration and Time Shift events
        for n, note in enumerate(track.notes[:-1]):
            # Pitch
            events.append(Event(
                name='Pitch',
                time=note.start,
                value=note.pitch,
                text=note.pitch))
            # Velocity
            events.append(Event(
                name='Velocity',
                time=note.start,
                value=note.velocity,
                text=f'{note.velocity}'))
            # Duration
            duration = note.end - note.start
            index = np.argmin(np.abs([ticks - duration for ticks in dur_bins]))
            events.append(Event(
                name='Duration',
                time=note.start,
                value='.'.join(map(str, self.durations[index])),
                text=f'{duration} ticks'))
            # Time-Shift
            time_shift = track.notes[n + 1].start - note.start
            index = np.argmin(np.abs([ticks - time_shift for ticks in dur_bins]))
            events.append(Event(
                name='Time-Shift',
                time=note.start,
                value='.'.join(map(str, self.durations[index])) if time_shift != 0 else '0.0.1',
                text=f'{time_shift} ticks'))
        # Adds the last note
        if track.notes[-1].pitch not in self.pitch_range:
            if len(events) > 0:
                del events[-1]
        else:
            events.append(Event(name='Pitch', time=track.notes[-1].start, value=track.notes[-1].pitch,
                                text=track.notes[-1].pitch))
            events.append(Event(name='Velocity', time=track.notes[-1].start, value=track.notes[-1].velocity,
                                text=f'{track.notes[-1].velocity}'))
            duration = track.notes[-1].end - track.notes[-1].start
            index = np.argmin(np.abs([ticks - duration for ticks in dur_bins]))
            events.append(Event(name='Duration', time=track.notes[-1].start,
                                value='.'.join(map(str, self.durations[index])), text=f'{duration} ticks'))

        events.sort(key=lambda x: x.time)

        return self._events_to_tokens(events)

    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        """ Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and a "Dummy" tempo change
        """
        events = self._tokens_to_events(tokens)

        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        current_tick = 0
        count = 0

        while count < len(events):
            if events[count].name == 'Pitch':
                try:
                    if events[count + 1].name == 'Velocity' and events[count + 2].name == 'Duration':
                        pitch = int(events[count].value)
                        vel = int(events[count + 1].value)
                        beat, pos, res = map(int, events[count + 2].value.split('.'))
                        duration = (beat * res + pos) * time_division // res
                        instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        count += 3
                except IndexError as _:
                    count += 1
            elif events[count].name == 'Time-Shift':
                beat, pos, res = map(int, events[count].value.split('.'))
                current_tick += (beat * res + pos) * time_division // res  # time shift
                count += 1
            else:
                count += 1

        return instrument, [TempoChange(TEMPO, 0)]

    def _create_vocabulary(self, program_tokens: bool) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
        """ Create the tokens <-> event dictionaries
        These dictionaries are created arbitrary according to constants defined
        at the top of this file.
        Note that when using them (prepare_data method), there is no error-handling
        so you must be sure that every case is covered by the dictionaries.
        NOTE: token index 0 is often used as a padding index during training, it might
        be preferable to leave it as it to pad your batch sequences
        NOTE 2: the original Structured MIDI Encoding doesn't use Chords tokens as its
        purpose is to draw uniform token types transitions, you can still use them but
        it will "break" this property

        :param program_tokens: creates tokens for MIDI programs in the dictionary
        :return: the dictionaries, one for each translation
        """
        token_type_indices = {'Pad': [0]}
        event_to_token = {'PAD_None': 0}  # starting at 1, token 0 is for padding

        # PITCH
        token_type_indices['Pitch'] = list(range(len(event_to_token), len(event_to_token) + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Pitch_{i}'] = len(event_to_token)

        # VELOCITY
        token_type_indices['Velocity'] = list(range(len(event_to_token), len(event_to_token) + len(self.velocities)))
        for i in self.velocities:
            event_to_token[f'Velocity_{i}'] = len(event_to_token)

        # DURATION
        token_type_indices['Duration'] = list(range(len(event_to_token), len(event_to_token) + len(self.durations)))
        for i in range(0, len(self.durations)):
            event_to_token[f'Duration_{".".join(map(str, self.durations[i]))}'] = len(event_to_token)

        # TIME SHIFT
        # same as durations but with 0.0.1 (1, this value is not important)
        event_to_token['Time-Shift_0.0.1'] = len(event_to_token)
        token_type_indices['Time-Shift'] = list(range(len(event_to_token), len(event_to_token) + len(self.durations)+1))
        for i in range(0, len(self.durations)):
            event_to_token[f'Time-Shift_{".".join(map(str, self.durations[i]))}'] = len(event_to_token)

        # PROGRAM
        if program_tokens:
            token_type_indices['Program'] = list(range(len(event_to_token), len(event_to_token) + 129))
            for program in range(-1, 128):  # -1 is drums
                event_to_token[f'Program_{program}'] = len(event_to_token)

        event_to_token[len(event_to_token)] = 'SOS_None'
        event_to_token[len(event_to_token)] = 'EOS_None'
        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        dic = dict()

        if 'Program' in self.token_types_indices:
            dic['Program'] = ['Bar']

        dic['Pitch'] = ['Velocity']
        dic['Velocity'] = ['Duration']
        dic['Duration'] = ['Time-Shift']
        dic['Time-Shift'] = ['Pitch']

        return dic
