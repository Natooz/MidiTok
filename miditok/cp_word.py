""" MIDI encoding method, similar to Compound Word
https://arxiv.org/abs/2101.02402

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Vocabulary, Event, detect_chords
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
    :param sos_eos_tokens: Adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """

    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 program_tokens: bool = PROGRAM_TOKENS, sos_eos_tokens: bool = False, params=None):
        # Indexes of additional token types within a compound token
        self.chord_idx = -3 if additional_tokens['Tempo'] and additional_tokens['Rest'] else -2 if \
            additional_tokens['Tempo'] or additional_tokens['Rest'] else -1
        self.rest_idx = -2 if additional_tokens['Tempo'] else -1
        self.tempo_idx = -1
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens,
                         {'program_tokens': program_tokens, 'sos_eos_tokens': sos_eos_tokens}, params)

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
                        tokens.append(self.create_cp_token(int(note.start), rest=f'{rest_beat}.0', desc='Rest'))
                        previous_tick += rest_beat * self.current_midi_metadata['time_division']

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min([r[1] for r in self.rests], key=lambda x: abs(x - rest_pos))
                        tokens.append(self.create_cp_token(int(note.start), rest=f'0.{rest_pos_temp}', desc='Rest'))
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
                    tokens.append(self.create_cp_token((current_bar + i + 1) * ticks_per_bar, bar=True, desc='Bar'))
                current_bar += nb_new_bars

                # Position
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                tokens.append(self.create_cp_token(int(note.start), pos=pos_index,
                                                   tempo=current_tempo if self.additional_tokens['Tempo'] else None,
                                                   desc='Position'))
                previous_tick = note.start

            # Note
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in
                                          self.durations_ticks[self.current_midi_metadata['time_division']]]))
            dur_value = '.'.join(map(str, self.durations[dur_index]))
            tokens.append(self.create_cp_token(int(note.start), pitch=note.pitch, vel=note.velocity, dur=dur_value,
                                               desc=f'{duration} ticks'))
            previous_note_end = max(previous_note_end, note.end)

        tokens.sort(key=lambda x: x[0].time)

        # Adds chord tokens if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            chord_events = detect_chords(track.notes, self.current_midi_metadata['time_division'], self._first_beat_res)
            count = 0
            for chord_event in chord_events:
                for e, cp_token in enumerate(tokens[count:]):
                    if cp_token[0].time == chord_event.time and cp_token[0].desc == 'Position':
                        cp_token[5] = self.vocab.event_to_token[f'Chord_{chord_event.value}']
                        count = e
                        break

        # Convert the first element of each compound token from Event to int
        for cp_token in tokens:
            cp_token[0] = self.vocab.event_to_token[f'Family_{cp_token[0].value}']

        return tokens

    def create_cp_token(self, time: int, bar: bool = False, pos: int = None, pitch: int = None, vel: int = None,
                        dur: str = None, chord: str = None, rest: str = None, tempo: int = None, program: int = None,
                        desc: str = '') -> List[Union[Event, int]]:
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
        :param desc: an optional argument for debug and used to spot position tokens in track_to_tokens
        :return: The compound token as a list of integers
        """
        cp_token_template = [Event(type_='Family', time=time, value='Metric', desc=desc),
                             self.vocab.event_to_token['Position_Ignore'],
                             self.vocab.event_to_token['Pitch_Ignore'],
                             self.vocab.event_to_token['Velocity_Ignore'],
                             self.vocab.event_to_token['Duration_Ignore']]
        if self.additional_tokens['Chord']:
            cp_token_template.append(self.vocab.event_to_token['Chord_Ignore'])
        if self.additional_tokens['Rest']:
            cp_token_template.append(self.vocab.event_to_token['Rest_Ignore'])
        if self.additional_tokens['Tempo']:
            cp_token_template.append(self.vocab.event_to_token['Tempo_Ignore'])

        if bar:
            cp_token_template[1] = self.vocab.event_to_token['Bar_None']
        elif pos is not None:
            cp_token_template[1] = self.vocab.event_to_token[f'Position_{pos}']
            if chord is not None:
                cp_token_template[self.chord_idx] = self.vocab.event_to_token[f'Chord_{chord}']
            if tempo is not None:
                cp_token_template[self.tempo_idx] = self.vocab.event_to_token[f'Tempo_{tempo}']
        elif rest is not None:
            cp_token_template[self.rest_idx] = self.vocab.event_to_token[f'Rest_{rest}']
        elif pitch is not None:
            cp_token_template[0].value = 'Note'
            cp_token_template[2] = self.vocab.event_to_token[f'Pitch_{pitch}']
            cp_token_template[3] = self.vocab.event_to_token[f'Velocity_{vel}']
            cp_token_template[4] = self.vocab.event_to_token[f'Duration_{dur}']
        elif program is not None:  # Exception here, the first element returned is an int
            cp_token_template[0] = self.vocab.event_to_token['Family_Program']
            cp_token_template[1] = self.vocab.event_to_token[f'Program_{program}']

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
            token_family = compound_token[0].value
            if token_family == 'Note':
                pitch = int(compound_token[2].value)
                vel = int(compound_token[3].value)
                beat, pos, res = map(int, compound_token[4].value.split('.'))
                duration = (beat * res + pos) * time_division // res
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
            elif token_family == 'Metric':
                if compound_token[1].type == 'Bar':
                    current_bar += 1
                    current_tick = current_bar * ticks_per_bar
                elif compound_token[1].value != 'Ignore':  # i.e. its a position
                    current_tick = current_bar * ticks_per_bar + int(compound_token[1].value) * ticks_per_sample
                    if self.additional_tokens['Tempo']:
                        tempo = int(compound_token[-1].value)
                        if tempo != tempo_changes[-1].tempo:
                            tempo_changes.append(TempoChange(tempo, current_tick))
                elif compound_token[self.rest_idx].value != 'Ignore':  # i.e. its a rest
                    beat, pos = map(int, compound_token[self.rest_idx].value.split('.'))
                    current_tick += beat * time_division + pos * ticks_per_sample
                    current_bar = current_tick // ticks_per_bar
        if len(tempo_changes) > 1:
            del tempo_changes[0]
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, program_tokens: bool, sos_eos_tokens: bool = False) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param program_tokens: will include tokens for MIDI programs
        :param sos_eos_tokens: will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        vocab = Vocabulary({'PAD_None': 0, 'Bar_None': 1, 'Family_Note': 2, 'Family_Metric': 3})

        # PITCH
        vocab.add_event('Pitch_Ignore')
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)

        # VELOCITY
        vocab.add_event('Velocity_Ignore')
        vocab.add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        vocab.add_event('Duration_Ignore')
        vocab.add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab.add_event('Position_Ignore')
        vocab.add_event(f'Position_{i}' for i in range(nb_positions))

        # CHORD
        if self.additional_tokens['Chord']:
            vocab.add_event('Chord_Ignore')
            vocab.add_event(f'Chord_{i}' for i in range(3, 6))  # non recognized chords (between 3 and 5 notes only)
            vocab.add_event(f'Chord_{chord_quality}' for chord_quality in CHORD_MAPS)

        # REST
        if self.additional_tokens['Rest']:
            vocab.add_event('Rest_Ignore')
            vocab.add_event(f'Rest_{".".join(map(str, rest))}' for rest in self.rests)

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.add_event('Tempo_Ignore')
            vocab.add_event(f'Tempo_{i}' for i in self.tempos)

        # PROGRAM
        if program_tokens:
            vocab.add_event('Family_Program')
            vocab.add_event(f'Program_{program}' for program in range(-1, 128))

        # SOS & EOS
        if sos_eos_tokens:
            vocab.add_sos_eos_to_vocab()

        return vocab

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        """ As with CP the tokens types are "merged", each state here corresponds to
        a "compound" token, which is characterized by the token types Program, Bar,
        Position/Chord/Tempo and Pitch/Velocity/Duration

        :return: the token types transitions dictionary
        """
        dic = dict()

        try:
            _ = self.vocab.tokens_of_type('Program')
            dic['Program'] = ['Bar']
        except KeyError:
            pass

        dic['Bar'] = ['Position']
        dic['Position'] = ['Pitch']
        dic['Pitch'] = ['Pitch', 'Bar', 'Position']
        if self.additional_tokens['Rest']:
            dic['Rest'] = ['Rest', 'Position']
            dic['Pitch'] += ['Rest']

        return dic
