""" Modified version of Octuple with no Program (Track) tokens
To use mainly for tasks handling a single track.

"""

from math import ceil
import json
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer
from .constants import *


class OctupleMonoEncoding(MIDITokenizer):
    """ Modified version of Octuple with no Program (Track) tokens
    To use mainly for tasks handling a single track.

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
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 program_tokens: bool = PROGRAM_TOKENS, params=None):
        additional_tokens['Chord'] = False  # Incompatible additional token
        additional_tokens['Empty'] = False  # could be done with special tokens for pitch/velocity/duration
        # used in place of positional encoding
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def save_params(self, out_dir: Union[str, Path, PurePath]):
        """ Override the parent class method to include additional parameter drum pitch range
        Saves the base parameters of this encoding in a txt file
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy

        :param out_dir: output directory to save the file
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(PurePath(out_dir, 'config').with_suffix(".txt"), 'w') as outfile:
            json.dump({'pitch_range': (self.pitch_range.start, self.pitch_range.stop),
                       'beat_res': {f'{k1}_{k2}': v for (k1, k2), v in self.beat_res.items()},
                       'nb_velocities': len(self.velocity_bins),
                       'additional_tokens': self.additional_tokens, 'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding},
                      outfile)

    def track_to_tokens(self, track: Instrument) -> List[List[int]]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            4: Position
            5: Bar
            (6: Tempo)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_frame = self.current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self.current_midi_metadata['time_division'] * 4

        # Check bar embedding limit, update if needed
        nb_bars = ceil(max(note.end for note in track.notes) / (self.current_midi_metadata['time_division'] * 4))
        if self.max_bar_embedding < nb_bars:
            count = len(self.event2token)
            for i in range(self.max_bar_embedding, nb_bars):
                self.event2token[f'Bar_{i}'] = count
                self.token2event[count] = f'Bar_{i}'
                self.token_types_indices['Bar'] += [count]
                count += 1
            self.max_bar_embedding = nb_bars

        tokens = []
        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        current_tempo = (np.abs(self.tempo_bins - current_tempo)).argmin()
        for note in track.notes:
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue

            # Positions and bars
            if note.start != current_tick:
                pos_index = int((note.start % ticks_per_bar) / ticks_per_frame)
                current_tick = note.start
                current_bar = current_tick // ticks_per_bar
                current_pos = pos_index

            # Note attributes
            velocity_index = (np.abs(self.velocity_bins - note.velocity)).argmin()
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in
                                          self.durations_ticks[self.current_midi_metadata['time_division']]]))
            token_ts = [self.event2token[f'Pitch_{note.pitch}'],
                        self.event2token[f'Velocity_{velocity_index}'],
                        self.event2token[f'Duration_{".".join(map(str, self.durations[dur_index]))}'],
                        self.event2token[f'Position_{current_pos}'],
                        self.event2token[f'Bar_{current_bar}']]

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
                token_ts.append(self.event2token[f'Tempo_{current_tempo}'])

            tokens.append(token_ts)

        return tokens

    def tokens_to_track(self, tokens: List[List[int]], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        """ Converts a sequence of tokens into a track object
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            4: Position
            5: Bar
            (6: Tempo)

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        events = [self.tokens_to_events(time_step) for time_step in tokens]

        ticks_per_frame = time_division // max(self.beat_res.values())
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)

        if self.additional_tokens['Tempo']:
            tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below
        else:  # default
            tempo_changes = [TempoChange(TEMPO, 0)] * 2  # the first will be deleted at the end of the method

        for time_step in events:
            # Note attributes
            pitch = int(time_step[0].value)
            vel = int(self.velocity_bins[int(time_step[1].value)])
            beat, pos, res = map(int, time_step[2].value.split('.'))
            duration = (beat * res + pos) * time_division // res

            # Time and track values
            current_pos = int(time_step[3].value)
            current_bar = int(time_step[4].value)
            current_tick = current_bar * time_division * 4 + current_pos * ticks_per_frame

            # Append the created note
            instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))

            # Tempo, adds a TempoChange if necessary
            if self.additional_tokens['Tempo']:
                tempo = int(self.tempo_bins[int(time_step[-1].value)])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))

        # Tempos
        del tempo_changes[0]
        return instrument, tempo_changes

    def _create_vocabulary(self, program_tokens) -> Tuple[dict, dict, dict]:
        """ Create the tokens <-> event dictionaries
        These dictionaries are created arbitrary according to constants defined
        at the top of this file.
        Note that when using them (prepare_data method), there is no error-handling
        so you must be sure that every case is covered by the dictionaries.
        NOTE: token index 0 is often used as a padding index during training, it might
        be preferable to leave it as it to pad your batch sequences
        NOTE 2: in this version Octuple, we still offer the possibility to create vocabulary
        with Program tokens, but these programs will not be included in the "merged" tokens
        of each note. Instead you can use them at the beginning of a sequence to indicate to
        a model the instrument.

        :param program_tokens: creates tokens for MIDI programs in the dictionary
        :return: the dictionaries, one for each translation
        """
        event_to_token = {'PAD_None': 0}  # token 0 for padding
        token_type_indices = {'Pad': [0]}
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

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        token_type_indices['Position'] = list(range(count, count + nb_positions))
        for i in range(0, nb_positions):  # position embeddings (positional encoding)
            event_to_token[f'Position_{i}'] = count
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

        # BAR --- MUST BE LAST IN DIC AS THIS MIGHT BE INCREASED
        token_type_indices['Bar'] = list(range(count, count + self.max_bar_embedding))
        for i in range(self.max_bar_embedding):  # bar embeddings (positional encoding)
            event_to_token[f'Bar_{i}'] = count
            count += 1

        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        return {}  # not relevant for this encoding
