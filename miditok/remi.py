""" MIDI encoding method, similar to the REMI introduced in the Pop Music Transformer paper
https://arxiv.org/abs/2002.00212

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords
from .constants import *


class REMIEncoding(MIDITokenizer):
    """ MIDI encoding method, similar to the REMI introduced in the Pop Music Transformer paper
    https://arxiv.org/abs/2002.00212

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in frames per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, empty bars, tempo...)
    :param program_tokens: will add entries for MIDI programs in the dictionary, to use
            in the case of multitrack generation for instance
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
                 program_tokens: bool = PROGRAM_TOKENS, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def track_to_tokens(self, track: Instrument) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_frame = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self.current_midi_metadata['time_division'] * 4
        events = []

        # Creates Bar and Position events
        bar_ticks = np.arange(0, max(n.end for n in track.notes) + ticks_per_bar, ticks_per_bar)
        for t, tick in enumerate(bar_ticks):  # creating a "Bar" event at each beginning of bars
            events.append(Event(
                name='Bar',
                time=tick,
                value=None,
                text=t))

            if self.additional_tokens['Empty']:
                # We consider a note inside a bar when its onset time is within the bar
                # as it is how the note messages will be put in the sequence
                notes_in_this_bar = [note for note in track.notes if tick <= note.start < tick + ticks_per_bar]
                if len(notes_in_this_bar) == 0:
                    events.append(Event(  # marks an empty bar
                        name='Empty',
                        time=tick,
                        value=None,
                        text=t))

        # Creates the Pitch, Velocity and Duration events
        current_tick = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        current_tempo = (np.abs(self.tempo_bins - current_tempo)).argmin()
        for note in track.notes:
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            # Position
            if note.start != current_tick:
                pos_index = int((note.start % ticks_per_bar) / ticks_per_frame)
                events.append(Event(
                    name='Position',
                    time=note.start,
                    value=pos_index,
                    text=note.start))
                current_tick = note.start
                # (Tempo)
                if self.additional_tokens['Tempo']:
                    # If the current tempo is not the last one
                    if current_tempo_idx + 1 < len(self.current_midi_metadata['tempo_changes']):
                        # Will loop over incoming tempo changes
                        for tempo_change in self.current_midi_metadata['tempo_changes'][current_tempo_idx + 1:]:
                            # If this tempo change happened before the current moment
                            if tempo_change.time <= current_tick:
                                current_tempo = (np.abs(self.tempo_bins - tempo_change.tempo)).argmin()
                                current_tempo_idx += 1  # update tempo value (might not change) and index
                            elif tempo_change.time > current_tick:
                                break  # this tempo change is beyond the current time step, we break the loop
                    events.append(Event(
                        name='Tempo',
                        time=note.start,
                        value=current_tempo,
                        text=note.start))
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
            index = np.argmin(np.abs([ticks - duration for ticks in
                                      self.durations_ticks[self.current_midi_metadata['time_division']]]))
            events.append(Event(
                name='Duration',
                time=note.start,
                value='.'.join(map(str, self.durations[index])),
                text=f'{duration} ticks'))

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            events += detect_chords(track.notes, self.current_midi_metadata['time_division'])

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

        ticks_per_frame = time_division // max(self.beat_res.values())
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        if self.additional_tokens['Tempo']:
            tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below
        else:  # default
            tempo_changes = [TempoChange(TEMPO, 0)] * 2  # the first will be deleted at the end of the method

        current_tick = 0
        current_bar = -1
        for ei, event in enumerate(events):
            if event.name == 'Bar':
                current_bar += 1
                current_tick = current_bar * time_division * 4
            elif event.name == 'Position':
                current_tick = current_bar * time_division * 4 + int(event.value) * ticks_per_frame
            elif event.name == 'Tempo':
                # If your encoding include tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                tempo = int(self.tempo_bins[int(event.value)])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            elif event.name == 'Pitch':
                try:
                    if events[ei + 1].name == 'Velocity' and events[ei + 2].name == 'Duration':
                        pitch = int(events[ei].value)
                        vel = int(self.velocity_bins[int(events[ei + 1].value)])
                        beat, pos, res = map(int, events[ei + 2].value.split('.'))
                        duration = (beat * res + pos) * time_division // res
                        instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                except IndexError as _:  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished
        del tempo_changes[0]
        return instrument, tempo_changes

    def _create_vocabulary(self, program_tokens: bool) -> Tuple[dict, dict, dict]:
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
        event_to_token = {'PAD_None': 0, 'Bar_None': 1}  # starting at 1, token 0 for padding
        token_type_indices = {'Pad': [0], 'Bar': [1]}
        count = 2
        if self.additional_tokens['Empty']:
            event_to_token['Empty_None'] = count
            token_type_indices['Empty'] = [count]
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
        token_type_indices['Position'] = list(range(count, count + nb_positions))
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

        # TEMPO
        if self.additional_tokens['Tempo']:
            token_type_indices['Tempo'] = list(range(count, count + len(self.tempo_bins)))
            for i in range(len(self.tempo_bins)):
                event_to_token[f'Tempo_{i}'] = count
                count += 1

        # PROGRAM
        if program_tokens:
            token_type_indices['Program'] = list(range(count, count + 129))
            for program in range(-1, 128):  # -1 is drums
                event_to_token[f'Program_{program}'] = count
                count += 1

        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        dic = dict()

        if 'Program' in self.token_types_indices:
            dic['Program'] = ['Bar']

        dic['Bar'] = ['Position']
        if self.additional_tokens['Empty']:
            dic['Bar'] += ['Empty']
            dic['Empty'] = ['Bar']

        dic['Position'] = ['Pitch']
        dic['Pitch'] = ['Velocity']
        dic['Velocity'] = ['Duration']
        dic['Duration'] = ['Pitch', 'Position', 'Bar']

        if self.additional_tokens['Chord']:
            dic['Chord'] = ['Pitch']
            dic['Duration'] += ['Chord']
            dic['Position'] += ['Chord']

        if self.additional_tokens['Tempo']:
            dic['Tempo'] = ['Chord', 'Pitch']
            dic['Position'] += ['Tempo']

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.name == "Program":
            return 0
        elif x.name == "Bar":
            return 1
        elif x.name == "Position" or x.name == "Empty":
            return 2
        elif x.name == "Chord" or x.name == "Tempo":  # actually tempos will be before chords
            return 3
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
