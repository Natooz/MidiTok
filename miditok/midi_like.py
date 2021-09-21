""" MIDI-like encoding method similar to ???
Music Transformer:

"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords
from .constants import *


class MIDILikeEncoding(MIDITokenizer):
    """ Structured MIDI encoding method as using in the Piano Inpainting Application
    https://arxiv.org/abs/2107.05944
    The token types follows the specific pattern:
    Pitch -> Velocity -> Duration -> Time shift -> back to Pitch ...
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
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...)
    :param program_tokens: will add entries for MIDI programs in the dictionary, to use
            in the case of multitrack generation for instance
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 program_tokens: bool = PROGRAM_TOKENS, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def track_to_tokens(self, track: Instrument) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note On
            events.append(Event(name='Note-On', time=note.start, value=note.pitch, text=note.end))
            # Velocity
            events.append(Event(name='Velocity', time=note.start, value=note.velocity, text=f'{note.velocity}'))
            # Note Off
            events.append(Event(name='Note-Off', time=note.end, value=note.pitch, text=note.end))
        # Adds tempo events if specified
        if self.additional_tokens['Tempo']:
            for tempo_change in self.current_midi_metadata['tempo_changes']:
                events.append(Event(name='Tempo', time=tempo_change.time, value=tempo_change.tempo,
                                    text=tempo_change.tempo))

        # Sorts events in the good order
        events.sort(key=lambda x: x.time)

        # Time Shift
        previous_tick = 0
        previous_note_end = track.notes[0].start + 1
        for e, event in enumerate(events.copy()):
            if event.name == 'Note-On':
                previous_note_end = max(previous_note_end, event.text)

            if event.time == previous_tick:
                continue

            time_shift = event.time - previous_tick

            if self.additional_tokens['Rest'] and event.time > previous_note_end:
                rest_beat, rest_pos = divmod(time_shift, self.current_midi_metadata['time_division'])
                rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                rest_pos = round(rest_pos / ticks_per_sample)
                if rest_beat > 0:
                    events.append(Event(name='Rest', time=previous_tick, value=f'{rest_beat}.0',
                                        text=f'{rest_beat}.0'))
                    previous_tick += rest_beat * self.current_midi_metadata['time_division']

                while rest_pos >= self.rests[0][1]:
                    rest_pos_temp = min([r[1] for r in self.rests], key=lambda x: abs(x - rest_pos))
                    events.append(Event(name='Rest', time=previous_tick, value=f'0.{rest_pos_temp}',
                                        text=f'0.{rest_pos_temp}'))
                    previous_tick += round(rest_pos_temp * ticks_per_sample)
                    rest_pos -= rest_pos_temp

            else:
                index = np.argmin(np.abs([ticks - time_shift for ticks in
                                          self.durations_ticks[self.current_midi_metadata['time_division']]]))
                events.append(Event(name='Time-Shift', time=previous_tick,
                                    value='.'.join(map(str, self.durations[index])), text=f'{time_shift} ticks'))
            previous_tick = event.time

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            events += detect_chords(track.notes, self.current_midi_metadata['time_division'], self._first_beat_res)

        events.sort(key=lambda x: (x.time, self._order(x)))

        return self._events_to_tokens(events)

    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False), default_duration: int = None) \
            -> Tuple[Instrument, List[TempoChange]]:
        """ Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration (in ticks) in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        events = self._tokens_to_events(tokens)

        max_duration = (self.durations[-1][0] + self.durations[-1][1]) * time_division
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].name == 'Note-On':
                try:
                    if events[ei + 1].name == 'Velocity':
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(ei+1, len(events)):
                            if events[i].name == 'Note-Off' and int(events[i].value) == pitch:
                                duration = offset_tick
                                break
                            elif events[i].name == 'Time-Shift':
                                beat, pos, res = map(int, events[i].value.split('.'))
                                offset_tick += (beat * res + pos) * time_division // res
                            elif events[ei].name == 'Rest':
                                beat, pos = map(int, events[ei].value.split('.'))
                                current_tick += beat * time_division + pos * ticks_per_sample
                            if offset_tick > max_duration:  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        ei += 1
                except IndexError as _:
                    pass
            elif events[ei].name == 'Time-Shift':
                beat, pos, res = map(int, events[ei].value.split('.'))
                current_tick += (beat * res + pos) * time_division // res
            elif events[ei].name == 'Rest':
                beat, pos = map(int, events[ei].value.split('.'))
                current_tick += beat * time_division + pos * ticks_per_sample
            elif events[ei].name == 'Tempo':
                tempo = int(events[ei].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, program_tokens: bool) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
        """ Create the tokens <-> event dictionaries
        These dictionaries are created arbitrary according to constants defined
        at the top of this file.
        Note that when using them (prepare_data method), there is no error-handling
        so you must be sure that every case is covered by the dictionaries.
        NOTE: token index 0 is often used as a padding index during training, it might
        be preferable to leave it as it

        :param program_tokens: creates tokens for MIDI programs in the dictionary
        :return: the dictionaries, one for each translation
        """
        token_type_indices = {'Pad': [0]}
        event_to_token = {'PAD_None': 0}  # starting at 1, token 0 is for padding

        # NOTE ON
        token_type_indices['Note-On'] = list(range(len(event_to_token), len(event_to_token) + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Note-On_{i}'] = len(event_to_token)

        # NOTE OFF
        token_type_indices['Note-Off'] = list(range(len(event_to_token), len(event_to_token) + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Note-Off_{i}'] = len(event_to_token)

        # VELOCITY
        token_type_indices['Velocity'] = list(range(len(event_to_token), len(event_to_token) + len(self.velocities)))
        for i in self.velocities:
            event_to_token[f'Velocity_{i}'] = len(event_to_token)

        # TIME SHIFTS
        token_type_indices['Time-Shift'] = list(range(len(event_to_token), len(event_to_token) + len(self.durations)))
        for i in range(0, len(self.durations)):
            event_to_token[f'Time-Shift_{".".join(map(str, self.durations[i]))}'] = len(event_to_token)

        # CHORD
        if self.additional_tokens['Chord']:
            token_type_indices['Chord'] = list(range(len(event_to_token), len(event_to_token) + 3 + len(CHORD_MAPS)))
            for i in range(3, 6):  # non recognized chords, just considers the nb of notes (between 3 and 5 only)
                event_to_token[f'Chord_{i}'] = len(event_to_token)
            for chord_quality in CHORD_MAPS:  # classed chords
                event_to_token[f'Chord_{chord_quality}'] = len(event_to_token)

        # REST
        if self.additional_tokens['Rest']:
            token_type_indices['Rest'] = list(range(len(event_to_token), len(event_to_token) + len(self.rests)))
            for i in range(0, len(self.rests)):
                event_to_token[f'Rest_{".".join(map(str, self.rests[i]))}'] = len(event_to_token)

        # TEMPO
        if self.additional_tokens['Tempo']:
            token_type_indices['Tempo'] = list(range(len(event_to_token), len(event_to_token) + len(self.tempos)))
            for i in self.tempos:
                event_to_token[f'Tempo_{i}'] = len(event_to_token)

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
            dic['Program'] = ['Note-On', 'Time-Shift']

        dic['Note-On'] = ['Velocity']
        dic['Velocity'] = ['Note-On', 'Time-Shift']
        dic['Time-Shift'] = ['Note-Off', 'Note-On']
        dic['Note-Off'] = ['Note-Off', 'Note-On', 'Time-Shift']

        if self.additional_tokens['Chord']:
            dic['Chord'] = ['Note-On']
            dic['Time-Shift'] += ['Chord']
            dic['Note-Off'] += ['Chord']

        if self.additional_tokens['Tempo']:
            dic['Time-Shift'] += ['Tempo']
            dic['Tempo'] = ['Note-On']
            if self.additional_tokens['Chord']:
                dic['Tempo'] += ['Chord']

        if self.additional_tokens['Rest']:
            dic['Rest'] = ['Rest', 'Note-On']
            if self.additional_tokens['Chord']:
                dic['Rest'] += ['Chord']
            dic['Note-Off'] = ['Rest']

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.name == 'Program':
            return 0
        elif x.name == 'Note-Off':
            return 1
        elif x.name == 'Tempo':
            return 2
        elif x.name == "Chord":
            return 3
        elif x.name == 'Time-Shift' or x.name == 'Rest':
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
