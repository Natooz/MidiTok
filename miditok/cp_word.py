""" MIDI encoding method, similar to Compound Word
https://arxiv.org/abs/2101.02402

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords
from .constants import *


class CPWordEncoding(MIDITokenizer):
    """ MIDI encoding method, similar to Compound Word
    https://arxiv.org/abs/2101.02402
    Each compound token will be a list of the form:
        (index. Token type)
        0. Family
        1. Bar/Position
        2. Pitch
        3. Velocity
        4. Duration
        (5. Chord) (optionals, chords occurring with position tokens)
        (6. Tempo) optional, occurring with position tokens
    This means a "compound token" can contain between 5 to 7 elements depending on
    your encoding parameters (additional tokens).
    (the choice of using indexes instead of dictionary with keys is to reduce the memory
    and storage usage for saved token files)



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
        # Indexes of additional token types within a compound token
        self.chord_idx = -3 if additional_tokens['Tempo'] and additional_tokens['Rest'] else -2 if \
            additional_tokens['Tempo'] or additional_tokens['Rest'] else -1
        self.rest_idx = -2 if additional_tokens['Tempo'] else -1
        self.tempo_idx = -1
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def track_to_tokens(self, track: Instrument) -> List[List[int]]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self.current_midi_metadata['time_division'] * 4
        min_rest = self.current_midi_metadata['time_division'] * (
                    self.rests[0][0] * self._first_beat_res + self.rests[0][1]) if self.additional_tokens['Rest'] else 0
        tokens = []  # list of lists of tokens

        # Creates tokens
        previous_tick = -1
        previous_note_end = track.notes[0].start + 1  # so that no rest is created before the first note
        current_bar = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        for note in track.notes:
            # Bar / Position / (Tempo) / (Rest)
            if note.start != previous_tick:

                # (Rest)
                if self.additional_tokens['Rest'] and note.start > previous_note_end and \
                        note.start - previous_tick > min_rest:
                    rest_beat, rest_pos = divmod(note.start - previous_tick,
                                                 self.current_midi_metadata['time_division'])
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        tokens.append(self.create_cp_token(int(note.start), rest=f'{rest_beat}.0', text='Rest'))
                        previous_tick += rest_beat * self.current_midi_metadata['time_division']

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min([r[1] for r in self.rests], key=lambda x: abs(x - rest_pos))
                        tokens.append(self.create_cp_token(int(note.start), rest=f'0.{rest_pos_temp}', text='Rest'))
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    current_bar = previous_tick // ticks_per_bar  # updates current bar value

                # (Tempo)
                if self.additional_tokens['Tempo']:
                    # If the current tempo is not the last one
                    if current_tempo_idx + 1 < len(self.current_midi_metadata['tempo_changes']):
                        # Will loop over incoming tempo changes
                        for tempo_change in self.current_midi_metadata['tempo_changes'][current_tempo_idx + 1:]:
                            # If this tempo change happened before the current moment
                            if tempo_change.time <= note.start:
                                current_tempo = tempo_change.tempo
                                current_tempo_idx += 1  # update tempo value (might not change) and index
                            elif tempo_change.time > note.start:
                                break  # this tempo change is beyond the current time step, we break the loop

                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    tokens.append(self.create_cp_token((current_bar + i + 1) * ticks_per_bar, bar=True, text='Bar'))
                current_bar += nb_new_bars

                # Position
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                tokens.append(self.create_cp_token(int(note.start), pos=pos_index,
                                                   tempo=current_tempo if self.additional_tokens['Tempo'] else None,
                                                   text='Position'))
                previous_tick = note.start

            # Note
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in
                                          self.durations_ticks[self.current_midi_metadata['time_division']]]))
            dur_value = '.'.join(map(str, self.durations[dur_index]))
            tokens.append(self.create_cp_token(int(note.start), pitch=note.pitch, vel=note.velocity, dur=dur_value,
                                               text=f'{duration} ticks'))
            previous_note_end = max(previous_note_end, note.end)

        tokens.sort(key=lambda x: x[0].time)

        # Adds chord tokens if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            chord_events = detect_chords(track.notes, self.current_midi_metadata['time_division'], self._first_beat_res)
            count = 0
            for chord_event in chord_events:
                for e, cp_token in enumerate(tokens[count:]):
                    if cp_token[0].time == chord_event.time and cp_token[0].text == 'Position':
                        cp_token[5] = self.event2token[f'Chord_{chord_event.value}']
                        count = e
                        break

        # Convert the first element of each compound token from Event to int
        for cp_token in tokens:
            cp_token[0] = self.event2token[f'Family_{cp_token[0].value}']

        return tokens

    def create_cp_token(self, time: int, bar: bool = False, pos: int = None, pitch: int = None, vel: int = None,
                        dur: str = None, chord: str = None, rest: str = None, tempo: int = None, program: int = None,
                        text: str = '') -> List[Union[Event, int]]:
        """ Create a CP Word token, with the following structure:
            (index. Token type)
            0. Family
            1. Bar/Position
            2. Pitch
            3. Velocity
            4. Duration
            (5. Chord) optional, chords occurring with position tokens
            (6. Rest) optional, rest acting as a time-shift event
            (7. Tempo) optional, occurring with position tokens
        NOTE: the first Family token (first in list) will be given as an Event object to keep track
        of time easily so that other method can sort CP tokens afterwards. Only exception is when
        creating a Program CP token (never done in MidiTok but implemented for you if needed).

        :param time: the current tick
        :param bar: True if this token represent a new bar occurring
        :param pos: the position index
        :param pitch: note pitch
        :param vel: note velocity
        :param dur: note duration
        :param chord: chord value
        :param rest: rest value
        :param tempo: tempo index
        :param program: a program number if you want to produce a Program CP token (read note above)
        :param text: an optional argument for debug and used to spot position tokens in track_to_tokens
        :return: The compound token as a list of integers
        """
        cp_token_template = [Event(name='Family', time=time, value='Metric', text=text),
                             self.event2token['BarPosition_Ignore'],
                             self.event2token['Pitch_Ignore'],
                             self.event2token['Velocity_Ignore'],
                             self.event2token['Duration_Ignore']]
        if self.additional_tokens['Chord']:
            cp_token_template.append(self.event2token['Chord_Ignore'])
        if self.additional_tokens['Rest']:
            cp_token_template.append(self.event2token['Rest_Ignore'])
        if self.additional_tokens['Tempo']:
            cp_token_template.append(self.event2token['Tempo_Ignore'])

        if bar:
            cp_token_template[1] = self.event2token['Bar_None']
        elif pos is not None:
            cp_token_template[1] = self.event2token[f'Position_{pos}']
            if chord is not None:
                cp_token_template[self.chord_idx] = self.event2token[f'Chord_{chord}']
            if tempo is not None:
                cp_token_template[self.tempo_idx] = self.event2token[f'Tempo_{tempo}']
        elif rest is not None:
            cp_token_template[self.rest_idx] = self.event2token[f'Rest_{rest}']
        elif pitch is not None:
            cp_token_template[0].value = 'Note'
            cp_token_template[2] = self.event2token[f'Pitch_{pitch}']
            cp_token_template[3] = self.event2token[f'Velocity_{vel}']
            cp_token_template[4] = self.event2token[f'Duration_{dur}']
        elif program is not None:  # Exception here, the first element returned is an int
            cp_token_template[0] = self.event2token['Family_Program']
            cp_token_template[1] = self.event2token[f'Program_{program}']

        return cp_token_template

    def tokens_to_track(self, tokens: List[List[int]], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        """ Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert time_division % max(self.beat_res.values()) == 0,\
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        events = [self._tokens_to_events(cp_token) for cp_token in tokens]

        ticks_per_sample = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below
        current_tick = 0
        current_bar = -1

        for compound_token in events:
            token_type = compound_token[0].value
            if token_type == 'Note':
                pitch = int(compound_token[2].value)
                vel = int(compound_token[3].value)
                beat, pos, res = map(int, compound_token[4].value.split('.'))
                duration = (beat * res + pos) * time_division // res
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
            elif token_type == 'Metric':
                if compound_token[1].name == 'Bar':
                    current_bar += 1
                    current_tick = current_bar * ticks_per_bar
                elif compound_token[1].name == 'Position':
                    current_tick = current_bar * ticks_per_bar + int(compound_token[1].value) * ticks_per_sample
                    if self.additional_tokens['Tempo']:
                        tempo = int(compound_token[-1].value)
                        if tempo != tempo_changes[-1].tempo:
                            tempo_changes.append(TempoChange(tempo, current_tick))
                elif compound_token[1].value == 'Ignore' and compound_token[self.rest_idx].value != 'Ignore':
                    beat, pos = map(int, compound_token[self.rest_idx].value.split('.'))
                    current_tick += beat * time_division + pos * ticks_per_sample
                    current_bar = current_tick // ticks_per_bar
        if len(tempo_changes) > 1:
            del tempo_changes[0]
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, program_tokens: bool) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
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
        token_type_indices = {'Pad': [0], 'BarPosition': [1], 'Family': [2, 3]}
        event_to_token = {'PAD_None': 0, 'Bar_None': 1, 'Family_Note': 2, 'Family_Metric': 3}

        # PITCH
        token_type_indices['Pitch'] = list(range(len(event_to_token), len(event_to_token) + len(self.pitch_range) + 1))
        event_to_token['Pitch_Ignore'] = len(event_to_token)
        for i in self.pitch_range:
            event_to_token[f'Pitch_{i}'] = len(event_to_token)

        # VELOCITY
        token_type_indices['Velocity'] = list(range(len(event_to_token), len(event_to_token) + len(self.velocities)+1))
        event_to_token['Velocity_Ignore'] = len(event_to_token)
        for i in self.velocities:
            event_to_token[f'Velocity_{i}'] = len(event_to_token)

        # DURATION
        token_type_indices['Duration'] = list(range(len(event_to_token), len(event_to_token) + len(self.durations) + 1))
        event_to_token['Duration_Ignore'] = len(event_to_token)
        for i in range(0, len(self.durations)):
            event_to_token[f'Duration_{".".join(map(str, self.durations[i]))}'] = len(event_to_token)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        token_type_indices['BarPosition'] += list(range(len(event_to_token), len(event_to_token) + nb_positions + 1))
        event_to_token['BarPosition_Ignore'] = len(event_to_token)
        for i in range(0, nb_positions):
            event_to_token[f'Position_{i}'] = len(event_to_token)

        # CHORD
        if self.additional_tokens['Chord']:
            token_type_indices['Chord'] = list(range(len(event_to_token), len(event_to_token) + 3 + len(CHORD_MAPS)+1))
            event_to_token['Chord_Ignore'] = len(event_to_token)
            for i in range(3, 6):  # non recognized chords, just considers the nb of notes (between 3 and 5 only)
                event_to_token[f'Chord_{i}'] = len(event_to_token)
            for chord_quality in CHORD_MAPS:  # classed chords
                event_to_token[f'Chord_{chord_quality}'] = len(event_to_token)

        # REST
        if self.additional_tokens['Rest']:
            token_type_indices['Rest'] = list(range(len(event_to_token), len(event_to_token) + len(self.rests) + 1))
            event_to_token['Rest_Ignore'] = len(event_to_token)
            for i in range(0, len(self.rests)):
                event_to_token[f'Rest_{".".join(map(str, self.rests[i]))}'] = len(event_to_token)

        # TEMPO
        if self.additional_tokens['Tempo']:
            token_type_indices['Tempo'] = list(range(len(event_to_token), len(event_to_token) + len(self.tempos) + 1))
            event_to_token['Tempo_Ignore'] = len(event_to_token)
            for i in self.tempos:
                event_to_token[f'Tempo_{i}'] = len(event_to_token)

        # PROGRAM
        if program_tokens:
            token_type_indices['Program'] = list(range(len(event_to_token), len(event_to_token) + 129))
            token_type_indices['Family'] += [len(event_to_token)]
            event_to_token['Family_Program'] = len(event_to_token)
            for program in range(-1, 128):  # -1 is drums
                event_to_token[f'Program_{program}'] = len(event_to_token)

        event_to_token[len(event_to_token)] = 'SOS_None'
        event_to_token[len(event_to_token)] = 'EOS_None'
        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        """ As with CP the tokens types are "merged", each state here corresponds to
        a "compound" token, which is characterized by the token types Program, Bar,
        Position/Chord/Tempo and Pitch/Velocity/Duration

        :return: the token types transitions dictionary
        """
        dic = dict()

        if 'Program' in self.token_types_indices:
            dic['Program'] = ['Bar']

        dic['Bar'] = ['Position']
        dic['Position'] = ['Pitch']
        dic['Pitch'] = ['Pitch', 'Bar', 'Position']
        if self.additional_tokens['Rest']:
            dic['Rest'] = ['Rest', 'Position']
            dic['Pitch'] += ['Rest']

        return dic
