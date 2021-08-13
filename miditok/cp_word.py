""" MIDI encoding method, similar to Compound Word
https://arxiv.org/abs/2101.02402

"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import Instrument, Note

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords
from .constants import *


class CPWordEncoding(MIDITokenizer):
    """ MIDI encoding method, similar to Compound Word
    https://arxiv.org/abs/2101.02402
    In this implementation the "Ignore" tokens are optional.
    If given, each compound token will be a list of the form:
        0. Family
        1. Bar/Position
        2. Pitch
        3. Velocity
        4. Duration
        5. Chord/Empty (optionals, chords occurring with positions, empty with bars)
        6. Tempo (optional, occurring with positions)
    This means a "compound token" can contain between 5 to 7 elements depending on
    your encoding parameters (additional tokens).
    (the choice of using indexes instead of dictionary with keys is to reduce the memory
    and storage usage for saved token files)

    With no "Ignore" tokens, the compound tokens (lists) will have the form:
    For Metrics:
        0. Family
        1. Bar/Position/Empty (Empty is optional)
        2. Chord (optional)
        3. Tempo (optional)
    and for Notes:
        0. Family
        1. Note
        2. Velocity
        3. Duration
    and Programs (that you must do yourself when creating your training samples):
        0. Family
        1. Program

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
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens)

    def events_to_tokens(self, events: List[List[Event]]) -> List[List[int]]:
        tokens = []
        for compound_token in events:
            tokens.append([self.event2token[f'{event.name}_{event.value}'] for event in compound_token])
        return tokens

    def tokens_to_events(self, tokens: List[List[int]]) -> List[List[Event]]:
        events = []
        for compound_token in tokens:
            time_step = []
            for token in compound_token:
                name, val = self.token2event[token].split('_')
                time_step.append(Event(name, None, val, None))
            events.append(time_step)
        return events

    def track_to_events(self, notes: List[Note], time_division: int, drum: Optional[bool] = False) -> List[List[Event]]:
        """ Converts a track (list of Note objects) into Event objects

        :param notes: notes of the track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param drum: specify if the notes treated are from a drum track (if it is the case no chord should be detected)
        :return: list of events
                 the events should be in the order Bar -> Position -> Chord -> Pitch -> Velocity -> Duration
        """
        if self.additional_tokens['Ignore']:
            return self._track_to_events_ignore(notes, time_division, drum)
        return self._track_to_events_no_ignore(notes, time_division, drum)

    def _track_to_events_ignore(self, notes: List[Note], time_division: int, drum: Optional[bool] = False) -> \
            List[List[Event]]:
        """ Converts a track (list of Note objects) into Event objects

        :param notes: notes of the track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param drum: specify if the notes treated are from a drum track (if it is the case no chord should be detected)
        :return: list of events
                 the events should be in the order Bar -> Position -> Chord -> Pitch -> Velocity -> Duration
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # it should have been done in quantization function
        ticks_per_frame = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        events = []  # list of lists of Event objects

        def create_event(time: int, bar=False, pos=None, pitch=None, vel=None, dur=None, chord=None, empty=False,
                         text='') -> List[Event]:
            event_template = [Event(name='Family', time=time, value='Metric', text=text),
                              Event(name='BarPosition', time=time, value='Ignore', text=text),
                              Event(name='Pitch', time=time, value='Ignore', text=text),
                              Event(name='Velocity', time=time, value='Ignore', text=text),
                              Event(name='Duration', time=time, value='Ignore', text=text)]
            if self.additional_tokens['Chord'] or self.additional_tokens['Empty']:
                event_template += [Event(name='ChordEmpty', time=time, value='Ignore', text=text)]

            if bar:
                event_template[1] = Event(name='Bar', time=time, value=None, text=text)
                if empty:
                    event_template[5] = Event(name='Empty', time=time, value=None, text=text)
            elif pos is not None:
                event_template[1] = Event(name='Position', time=time, value=pos, text=text)
            elif pitch is not None:
                event_template[0].value = 'Note'
                event_template[2].value = pitch
                event_template[3].value = vel
                event_template[4].value = dur

            if chord is not None:
                event_template[5] = Event(name='Chord', time=time, value=chord, text=text)

            return event_template

        # Creates Bar and Position events
        bar_ticks = np.arange(0, max(n.end for n in notes) + ticks_per_bar, ticks_per_bar)
        for t, tick in enumerate(bar_ticks):  # creating a "Bar" event at each beginning of bars
            emp = False
            if self.additional_tokens['Empty']:
                # We consider a note inside a bar when its onset time is within the bar
                # as it is how the note messages will be put in the sequence
                notes_in_this_bar = [note for note in notes if tick <= note.start < tick + ticks_per_bar]
                if len(notes_in_this_bar) == 0:
                    emp = True
            events.append(create_event(tick, bar=True, empty=emp, text=str(t)))

        # Creates the Pitch, Velocity and Duration events
        current_tick = -1
        for note in notes:
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            if note.start != current_tick:
                pos_index = int((note.start % ticks_per_bar) / ticks_per_frame)
                events.append(create_event(int(note.start), pos=pos_index, text=str(note.start)))
                current_tick = note.start

            # Note
            velocity_index = (np.abs(self.velocity_bins - note.velocity)).argmin()
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in self.durations_ticks[time_division]]))
            dur_value = '.'.join(map(str, self.durations[dur_index]))
            events.append(create_event(int(note.start), pitch=note.pitch, vel=velocity_index, dur=dur_value,
                                       text=f'{duration} ticks'))

        events.sort(key=lambda x: x[0].time)

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not drum:
            chord_events = detect_chords(notes, time_division)
            count = 0
            for chord_event in chord_events:
                for e, compound_token in enumerate(events[count:]):
                    if compound_token[1].time == chord_event.time and compound_token[1].name == 'Position':
                        compound_token[5] = chord_event
                        count = e
                        break

        return events

    def _track_to_events_no_ignore(self, notes: List[Note], time_division: int, drum: Optional[bool] = False) -> \
            List[List[Event]]:
        """ Converts a track (list of Note objects) into Event objects

        :param notes: notes of the track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param drum: specify if the notes treated are from a drum track (if it is the case no chord should be detected)
        :return: list of events
                 the events should be in the order Bar -> Position -> Chord -> Pitch -> Velocity -> Duration
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # it should have been done in quantization function
        ticks_per_frame = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        events = []  # list of lists of Event objects

        # Creates Bar and Position events
        bar_ticks = np.arange(0, max(n.end for n in notes) + ticks_per_bar, ticks_per_bar)
        for t, tick in enumerate(bar_ticks):  # creating a "Bar" event at each beginning of bars

            events.append([Event(name='Family', time=tick, value='Metric', text=t),
                           Event(name='Bar', time=tick, value=None, text=t)])

            if self.additional_tokens['Empty']:
                # We consider a note inside a bar when its onset time is within the bar
                # as it is how the note messages will be put in the sequence
                notes_in_this_bar = [note for note in notes if tick <= note.start < tick + ticks_per_bar]
                if len(notes_in_this_bar) == 0:
                    events[-1].append(Event(name='Empty', time=tick, value=None, text=t))

        # Creates the Pitch, Velocity and Duration events
        current_tick = -1
        for note in notes:
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            if note.start != current_tick:
                pos_index = int((note.start % ticks_per_bar) / ticks_per_frame)
                events.append([Event(name='Family', time=note.start, value='Metric', text=note.start),
                               Event(name='Position', time=note.start, value=pos_index, text=note.start)])

                current_tick = note.start

            # Note
            velocity_index = (np.abs(self.velocity_bins - note.velocity)).argmin()
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in self.durations_ticks[time_division]]))
            events.append([Event(name='Family', time=note.start, value='Note', text=note.start),
                           Event(name='Pitch', time=note.start, value=note.pitch, text=note.pitch),
                           Event(name='Velocity', time=note.start, value=velocity_index,
                                 text=f'{note.velocity}/{self.velocity_bins[velocity_index]}'),
                           Event(name='Duration', time=note.start, value='.'.join(map(str, self.durations[dur_index])),
                                 text=f'{duration} ticks')])

        events.sort(key=lambda x: x[0].time)

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not drum:
            chord_events = detect_chords(notes, time_division)
            count = 0
            for chord_event in chord_events:
                for e, compound_token in enumerate(events[count:]):
                    if compound_token[1].time == chord_event.time and compound_token[1].name == 'Position':
                        compound_token.append(chord_event)
                        count = e
                        break

        return events

    def events_to_track(self, events: List[List[Event]], time_division: int,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Instrument:
        """ Transform a list of Event objects into an instrument object

        :param events: list of Event objects to convert to a track
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object
        """
        ticks_per_frame = time_division // max(self.beat_res.values())

        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        current_tick = 0
        current_bar = -1
        offset = 1 if self.additional_tokens['Ignore'] else 0

        for compound_token in events:
            token_type = compound_token[0].value
            if token_type == 'Note':
                pitch = int(compound_token[1 + offset].value)
                vel = int(self.velocity_bins[int(compound_token[2 + offset].value)])
                beat, pos, res = map(int, compound_token[3 + offset].value.split('.'))
                duration = (beat * res + pos) * time_division // res
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
            elif token_type == 'Metric':
                if compound_token[1].name == 'Bar':
                    current_bar += 1
                    current_tick = current_bar * time_division * 4
                elif compound_token[1].name == 'Position':
                    current_tick = current_bar * time_division * 4 + int(compound_token[1].value) * ticks_per_frame

        return instrument

    def create_token_dicts(self, program_tokens: bool) -> Tuple[dict, dict, dict]:
        """ Create the tokens <-> event dictionaries
        These dictionaries are created arbitrary according to constants defined
        at the top of this file.
        Note that when using them (prepare_data method), there is no error-handling
        so you must be sure that every case is covered by the dictionaries.
        NOTE: token index 0 is often used as a padding index during training, it might
        be preferable to leave it as it to pad your batch sequences

        :param program_tokens: creates tokens for MIDI programs in the dictionary
        :return: the dictionaries, one for each translation
        """
        event_to_token = {'PAD_None': 0, 'Bar_None': 1, 'Family_Note': 3, 'Family_Metric': 4}
        token_type_indices = {'Pad': [0], 'BarPosition': [1], 'Family': [2, 3]}
        count = 5
        if self.additional_tokens['Empty']:  # will be considered in the Metric family
            event_to_token['Empty_None'] = count
            token_type_indices['ChordEmpty'] = [count]
            count += 1

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

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        token_type_indices['BarPosition'] += list(range(count, count + nb_positions))
        for i in range(0, nb_positions):
            event_to_token[f'Position_{i}'] = count
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

        # IGNORE
        if self.additional_tokens['Ignore']:
            event_to_token['BarPosition_Ignore'] = count
            token_type_indices['BarPosition'] += [count]
            event_to_token['Pitch_Ignore'] = count + 1
            token_type_indices['Pitch'] += [count + 1]
            event_to_token['Velocity_Ignore'] = count + 2
            token_type_indices['Velocity'] += [count + 2]
            event_to_token['Duration_Ignore'] = count + 3
            token_type_indices['Duration'] += [count + 3]
            count += 4
            if self.additional_tokens['Chord'] or self.additional_tokens['Empty']:
                event_to_token['ChordEmpty_Ignore'] = count
                try:
                    token_type_indices['ChordEmpty'] += [count]
                except KeyError as _:
                    token_type_indices['ChordEmpty'] = [count]
                count += 1

            token_type_indices['Empty'] = [count]
            count += 1

        # PROGRAM
        if program_tokens:
            event_to_token['Family_Program'] = count
            token_type_indices['Family'] += [count]
            count += 1
            token_type_indices['Program'] = list(range(count, count + 129))
            for program in range(-1, 128):  # -1 is drums
                event_to_token[f'Program_{program}'] = count
                count += 1

        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        """ As with CP the tokens types are "merged", each state here corresponds to
        a "compound" token, which is characterized by the token types Program, Bar,
        Position, Pitch and Empty

        :return: the token types transitions dictionary
        """
        dic = dict()

        if 'Program' in self.token_types_indices:
            dic['Program'] = ['Bar']

        dic['Bar'] = ['Position']
        dic['Position'] = ['Pitch']
        dic['Pitch'] = ['Pitch', 'Bar', 'Position']

        return dic
