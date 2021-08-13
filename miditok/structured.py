""" Structured MIDI encoding method as using in the Piano Inpainting Application
https://arxiv.org/abs/2107.05944

"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import Instrument, Note

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords
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
    NOTE: the original Structured MIDI Encoding doesn't use Chords tokens as its
    purpose is to draw uniform token types transitions, you can still use them but
    it will "break" this property

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in frames per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, empty bars, tempo...)
    :param program_tokens: will add entries for MIDI programs in the dictionary, to use
            in the case of multitrack generation for instance
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 program_tokens: bool = PROGRAM_TOKENS):
        # Incompatible additional tokens
        additional_tokens['Empty'] = False
        additional_tokens['Tempo'] = False
        additional_tokens['Ignore'] = False
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens)

    def track_to_events(self, notes: List[Note], time_division: int, drum: Optional[bool] = False) -> List[Event]:
        """ Converts a track (list of Note objects) into Event objects

        :param notes: notes of the track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param drum: specify if the notes treated are from a drum track (if it is the case no chord should be detected)
        :return: list of events
                 the events should be in the order Bar -> Position -> Chord -> Pitch -> Velocity -> Duration
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # it should have been done in quantization function
        events = []

        # First time shift if needed
        if notes[0].start != 0:
            if notes[0].start > max(self.durations_ticks[time_division]):
                time_shift = notes[0].start % time_division  # beat wise
            else:
                time_shift = notes[0].start
            index = np.argmin(np.abs([ticks - time_shift for ticks in self.durations_ticks[time_division]]))
            events.append(Event(
                name='Time-Shift',
                time=0,
                value='.'.join(map(str, self.durations[index])),
                text=f'{time_shift} ticks'))

        # Creates the Pitch, Velocity, Duration and Time Shift events
        for n, note in enumerate(notes[:-1]):
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            # Pitch
            events.append(Event(
                name='Pitch',
                time=note.start,
                value=note.pitch,
                text=note.pitch))
            # Velocity
            velocity_index = (np.abs(self.velocity_bins - note.velocity)).argmin()
            events.append(Event(
                name='Velocity',
                time=note.start,
                value=velocity_index,
                text=f'{note.velocity}/{self.velocity_bins[velocity_index]}'))
            # Duration
            duration = note.end - note.start
            index = np.argmin(np.abs([ticks - duration for ticks in self.durations_ticks[time_division]]))
            events.append(Event(
                name='Duration',
                time=note.start,
                value='.'.join(map(str, self.durations[index])),
                text=f'{duration} ticks'))
            # Time-Shift
            time_shift = notes[n + 1].start - note.start
            index = np.argmin(np.abs([ticks - time_shift for ticks in self.durations_ticks[time_division]]))
            events.append(Event(
                name='Time-Shift',
                time=note.start,
                value='.'.join(map(str, self.durations[index])) if time_shift != 0 else '0.0.1',
                text=f'{time_shift} ticks'))
        # Adds the last note
        events.append(Event(name='Pitch', time=notes[-1].start, value=notes[-1].pitch, text=notes[-1].pitch))
        velocity_index = (np.abs(self.velocity_bins - notes[-1].velocity)).argmin()
        events.append(Event(name='Velocity', time=notes[-1].start, value=velocity_index,
                            text=f'{notes[-1].velocity}/{self.velocity_bins[velocity_index]}'))
        duration = notes[-1].end - notes[-1].start
        index = np.argmin(np.abs([ticks - duration for ticks in self.durations_ticks[time_division]]))
        events.append(Event(name='Duration', time=notes[-1].start, value='.'.join(map(str, self.durations[index])),
                            text=f'{duration} ticks'))

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not drum:
            events += detect_chords(notes, time_division)

        events.sort(key=lambda x: x.time)

        return events

    def events_to_track(self, events: List[Event], time_division: int,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Instrument:
        """ Transform a list of Event objects into an instrument object

        :param events: list of Event objects to convert to a track
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object
        """
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        current_tick = 0
        count = 0

        while count < len(events):
            if events[count].name == 'Pitch':
                try:
                    if events[count + 1].name == 'Velocity' and events[count + 2].name == 'Duration':
                        pitch = int(events[count].value)
                        vel = int(self.velocity_bins[int(events[count + 1].value)])
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

        return instrument

    def create_token_dicts(self, program_tokens: bool) -> Tuple[dict, dict, dict]:
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
        event_to_token = {'PAD_None': 0}  # starting at 1, token 0 is for padding
        token_type_indices = {'Pad': [0]}  # Empty is for empty bars
        count = 1

        # PITCH
        token_type_indices['Pitch'] = list(range(count, count + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Pitch_{i}'] = count
            count += 1

        # VELOCITY
        token_type_indices['Velocity'] = list(range(count, count + len(self.velocity_bins)))
        for i in range(len(self.velocity_bins)):
            event_to_token[f'Velocity_{i}'] = count
            count += 1

        # DURATION
        token_type_indices['Duration'] = list(range(count, count + len(self.durations)))
        for i in range(0, len(self.durations)):
            event_to_token[f'Duration_{".".join(map(str, self.durations[i]))}'] = count
            count += 1

        # TIME SHIFT
        # same as durations but with 0.0.1 (1, this value is not important)
        event_to_token['Time-Shift_0.0.1'] = count
        count += 1
        token_type_indices['Time-Shift'] = list(range(count, count + len(self.durations) + 1))
        for i in range(0, len(self.durations)):
            event_to_token[f'Time-Shift_{".".join(map(str, self.durations[i]))}'] = count
            count += 1

        # CHORD
        if self.additional_tokens['Chord']:
            token_type_indices['Chord'] = list(range(count, count + 3 + len(CHORD_MAPS)))
            for i in range(3, 6):  # non recognized chords, just considers the nb of notes (between 3 and 5 only)
                event_to_token[f'Chord_{i}'] = count
                count += 1
            for chord_quality in CHORD_MAPS:  # classed chords
                event_to_token[f'Chord_{chord_quality}'] = count
                count += 1

        # PROGRAM
        if program_tokens:
            token_type_indices['Program'] = list(range(count, count + 129))
            for program in range(-1, 128):  # -1 is drums
                event_to_token[f'Program_{program}'] = count
                count += 1

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

        if self.additional_tokens['Chord']:
            dic['Chord'] = ['Pitch']
            dic['Time-Shift'] += ['Chord']

        return dic
