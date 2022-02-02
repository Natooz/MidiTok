""" MIDI encoding method, similar to the REMI introduced in the Pop Music Transformer paper
https://arxiv.org/abs/2002.00212

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Vocabulary, Event, detect_chords
from .constants import *


class REMI(MIDITokenizer):
    """ MIDI encoding method, similar to the REMI introduced in the Pop Music Transformer paper
    https://arxiv.org/abs/2002.00212

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...)
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """

    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
                 sos_eos_tokens: bool = False, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, params)

    def track_to_tokens(self, track: Instrument) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self.current_midi_metadata['time_division'] * 4
        dur_bins = self.durations_ticks[self.current_midi_metadata['time_division']]
        min_rest = self.current_midi_metadata['time_division'] * self.rests[0][0] + ticks_per_sample * self.rests[0][1]\
            if self.additional_tokens['Rest'] else 0

        events = []

        # Creates events
        previous_tick = -1
        previous_note_end = track.notes[0].start + 1  # so that no rest is created before the first note
        current_bar = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        for note in track.notes:
            if note.start != previous_tick:

                # (Rest)
                if self.additional_tokens['Rest'] and note.start > previous_note_end and \
                        note.start - previous_note_end >= min_rest:
                    previous_tick = previous_note_end
                    rest_beat, rest_pos = divmod(note.start-previous_tick, self.current_midi_metadata['time_division'])
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        events.append(Event(type_='Rest', time=previous_note_end, value=f'{rest_beat}.0',
                                            desc=f'{rest_beat}.0'))
                        previous_tick += rest_beat * self.current_midi_metadata['time_division']

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min([r[1] for r in self.rests], key=lambda x: abs(x - rest_pos))
                        events.append(Event(type_='Rest', time=previous_note_end, value=f'0.{rest_pos_temp}',
                                            desc=f'0.{rest_pos_temp}'))
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    current_bar = previous_tick // ticks_per_bar

                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    events.append(Event(type_='Bar', time=(current_bar + i + 1) * ticks_per_bar, value=None, desc=0))
                current_bar += nb_new_bars

                # Position
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                events.append(Event(type_='Position', time=note.start, value=pos_index, desc=note.start))

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
                            else:  # <==> elif tempo_change.time > previous_tick:
                                break  # this tempo change is beyond the current time step, we break the loop
                    events.append(Event(type_='Tempo', time=note.start, value=current_tempo, desc=note.start))

                previous_tick = note.start

            # Pitch / Velocity / Duration
            events.append(Event(type_='Pitch', time=note.start, value=note.pitch, desc=note.pitch))
            events.append(Event(type_='Velocity', time=note.start, value=note.velocity, desc=f'{note.velocity}'))
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(Event(type_='Duration', time=note.start, value='.'.join(map(str, self.durations[index])),
                                desc=f'{duration} ticks'))

            previous_note_end = max(previous_note_end, note.end)

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            events += detect_chords(track.notes, self.current_midi_metadata['time_division'], self._first_beat_res)

        events.sort(key=lambda x: (x.time, self._order(x)))

        return self.events_to_tokens(events)

    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        """ Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        events = self.tokens_to_events(tokens)

        ticks_per_sample = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        current_bar = -1
        previous_note_end = 0
        for ei, event in enumerate(events):
            if event.type == 'Bar':
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif event.type == 'Rest':
                beat, pos = map(int, events[ei].value.split('.'))
                if current_tick < previous_note_end:  # if in case successive rest happen
                    current_tick = previous_note_end
                current_tick += beat * time_division + pos * ticks_per_sample
                current_bar = current_tick // ticks_per_bar
            elif event.type == 'Position':
                if current_bar == -1:
                    current_bar = 0  # as this Position token occurs before any Bar token
                current_tick = current_bar * ticks_per_bar + int(event.value) * ticks_per_sample
            elif event.type == 'Tempo':
                # If your encoding include tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                tempo = int(event.value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            elif event.type == 'Pitch':
                try:
                    if events[ei + 1].type == 'Velocity' and events[ei + 2].type == 'Duration':
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)
                        duration = self._token_duration_to_ticks(events[ei + 2].value, time_division)
                        instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        previous_note_end = max(previous_note_end, current_tick + duration)
                except IndexError as _:  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished

        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, sos_eos_tokens: bool = False) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        vocab = Vocabulary({'PAD_None': 0, 'Bar_None': 1})

        # PITCH
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)

        # VELOCITY
        vocab.add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        vocab.add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab.add_event(f'Position_{i}' for i in range(nb_positions))

        # CHORD
        if self.additional_tokens['Chord']:
            vocab.add_event(f'Chord_{i}' for i in range(3, 6))  # non recognized chords (between 3 and 5 notes only)
            vocab.add_event(f'Chord_{chord_quality}' for chord_quality in CHORD_MAPS)

        # REST
        if self.additional_tokens['Rest']:
            vocab.add_event(f'Rest_{".".join(map(str, rest))}' for rest in self.rests)

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.add_event(f'Tempo_{i}' for i in self.tempos)

        # PROGRAM
        if self.additional_tokens['Program']:
            vocab.add_event(f'Program_{program}' for program in range(-1, 128))

        # SOS & EOS
        if sos_eos_tokens:
            vocab.add_sos_eos()

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        """ Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        try:
            _ = self.vocab.tokens_of_type('Program')
            dic['Program'] = ['Bar']
        except KeyError:
            pass

        dic['Bar'] = ['Position', 'Bar']

        dic['Position'] = ['Pitch']
        dic['Pitch'] = ['Velocity']
        dic['Velocity'] = ['Duration']
        dic['Duration'] = ['Pitch', 'Position', 'Bar']

        if self.additional_tokens['Chord']:
            dic['Chord'] = ['Pitch']
            dic['Duration'] += ['Chord']
            dic['Position'] += ['Chord']

        if self.additional_tokens['Tempo']:
            dic['Tempo'] = ['Chord', 'Pitch'] if self.additional_tokens['Chord'] else ['Pitch']
            dic['Position'] += ['Tempo']

        if self.additional_tokens['Rest']:
            dic['Rest'] = ['Rest', 'Position', 'Bar']
            dic['Duration'] += ['Rest']

        self._add_pad_type_to_graph(dic)
        return dic

    def token_types_errors(self, tokens: List[int], consider_pad: bool = False) -> float:
        """ Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a position token cannot have a value <= to the current position (it would go back in time)
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err = 0
        previous_type = self.vocab.token_type(tokens[0])
        current_pos = -1
        current_pitches = []

        def check(tok: int):
            nonlocal err, previous_type, current_pos, current_pitches
            token_type, token_value = self.vocab.token_to_event[tok].split('_')

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == 'Bar':  # reset
                    current_pos = -1
                    current_pitches = []
                elif token_type == 'Pitch':
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == 'Position':
                    if int(token_value) <= current_pos and previous_type != 'Rest':
                        err += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = []
            # Bad token type
            else:
                err += 1
            previous_type = token_type

        if consider_pad:
            for token in tokens[1:]:
                check(token)
        else:
            for token in tokens[1:]:
                if previous_type == 'PAD':
                    break
                check(token)
        return err / len(tokens)

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == 'Program':
            return 0
        elif x.type == 'Bar':
            return 1
        elif x.type == 'Position':
            return 2
        elif x.type == 'Chord' or x.type == 'Tempo':  # actually object_list will be before chords
            return 3
        elif x.type == 'Rest':
            return 5
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
