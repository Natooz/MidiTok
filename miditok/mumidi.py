""" MuMIDI encoding method, as introduced in PopMag
https://arxiv.org/abs/2008.07703

"""

from math import ceil
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event
from .utils import detect_chords
from .constants import PITCH_RANGE, NB_VELOCITIES, BEAT_RES, ADDITIONAL_TOKENS, TIME_DIVISION, TEMPO, \
    MIDI_INSTRUMENTS, CHORD_MAPS


# recommended range from the GM2 specs
# note: we ignore the "Applause" at pitch 88 of the orchestra drum set, increase to 89 if you need it
DRUM_PITCH_RANGE = range(27, 88)


class MuMIDI(MIDITokenizer):
    r"""MuMIDI encoding method, as introduced in PopMag
    https://arxiv.org/abs/2008.07703

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo)
    :param pad: will include a PAD token, used when training a model with batch of sequences of
            unequal lengths, and usually at index 0 of the vocabulary. (default: True)
    :param sos_eos: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary.
            (default: False)
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    :param drum_pitch_range: range of used MIDI pitches for drums exclusively
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 programs: List[int] = None, pad: bool = True, sos_eos: bool = False, mask: bool = False, params=None,
                 drum_pitch_range: range = DRUM_PITCH_RANGE):
        additional_tokens['Rest'] = False
        additional_tokens['TimeSignature'] = False  # not compatible
        self.drum_pitch_range = drum_pitch_range
        self.programs = list(range(-1, 128)) if programs is None else programs
        # used in place of positional encoding
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, pad, sos_eos, mask, True,
                         params=params)

    def save_params(self, out_path: Union[str, Path, PurePath], additional_attributes: Dict = None):
        r"""Overrides the parent class method to include additional parameter drum pitch range
        Saves the config / base parameters of the tokenizer in a file.
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy.
        NOTE: if you override this method, you should probably call it (super()) at the end
            and use the additional_attributes argument.
        NOTE 2: as json cant save tuples as keys, the beat ranges are saved as strings
        with the form startingBeat_endingBeat (underscore separating these two values)

        :param out_path: output path to save the file
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
        """
        if additional_attributes is None:
            additional_attributes = {}
        additional_attributes_tmp = {'max_bar_embedding': self.max_bar_embedding,
                                     'programs': self.programs,
                                     'drum_pitch_range': (self.drum_pitch_range.start, self.drum_pitch_range.stop),
                                     **additional_attributes}
        super().save_params(out_path, additional_attributes_tmp)

    def load_params(self, params: Union[str, Path, PurePath]):
        r"""Load parameters and set the encoder attributes

        :param params: can be a path to the parameter (json encoded) file
        """
        super().load_params(params)
        self.drum_pitch_range = range(*self.drum_pitch_range)

    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[int]]:
        r"""Override the parent class method
        Converts a MIDI file in a tokens representation, a sequence of "time steps".
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch / Position / Bar / Program / (Chord)
            (1: Velocity)
            (2: Duration)
            1 or 3: Current Bar embedding
            2 or 4: Current Position embedding
            (-1: Tempo)

        :param midi: the MIDI objet to convert
        :return: the token representation, i.e. tracks converted into sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self.durations_ticks:
            self.durations_ticks[midi.ticks_per_beat] = np.array([(beat * res + pos) * midi.ticks_per_beat // res
                                                                  for beat, pos, res in self.durations])
        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self.current_midi_metadata = {'time_division': midi.ticks_per_beat,
                                      'tempo_changes': midi.tempo_changes,
                                      'time_sig_changes': midi.time_signature_changes,
                                      'key_sig_changes': midi.key_signature_changes}

        # **************** OVERRIDE FROM HERE, KEEP THE LINES ABOVE IN YOUR METHOD ****************

        # Check bar embedding limit, update if needed
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.max_bar_embedding < nb_bars:
            self.vocab[4].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, nb_bars))
            self.max_bar_embedding = nb_bars

        # Convert each track to tokens
        note_tokens = []
        for track in midi.instruments:
            if track.program in self.programs:
                note_tokens += self.track_to_tokens(track)

        note_tokens.sort(key=lambda x: (x[0].time, x[0].desc))  # Sort by time then track

        ticks_per_sample = midi.ticks_per_beat / max(self.beat_res.values())
        ticks_per_bar = midi.ticks_per_beat * 4
        tokens = []

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track = -2  # because -2 doesn't exist
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        for note_event in note_tokens:
            # (Tempo) update tempo values current_tempo
            if self.additional_tokens['Tempo']:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(self.current_midi_metadata['tempo_changes']):
                    # Will loop over incoming tempo changes
                    for tempo_change in self.current_midi_metadata['tempo_changes'][current_tempo_idx + 1:]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note_event[0].time:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += 1  # update tempo value (might not change) and index
                        elif tempo_change.time > note_event[0].time:
                            break  # this tempo change is beyond the current time step, we break the loop
            # Positions and bars pos enc
            if note_event[0].time != current_tick:
                pos_index = int((note_event[0].time % ticks_per_bar) / ticks_per_sample)
                current_tick = note_event[0].time
                current_pos = pos_index
                current_track = -2  # reset
                # (New bar)
                if current_bar < current_tick // ticks_per_bar:
                    nb_new_bars = current_tick // ticks_per_bar - current_bar
                    for i in range(nb_new_bars):
                        bar_token = [self.vocab[0].event_to_token['Bar_None'],
                                     self.vocab[3].event_to_token['Position_Ignore'],
                                     self.vocab[4].event_to_token[f'Bar_{current_bar + i + 1}']]
                        if self.additional_tokens['Tempo']:
                            bar_token.append(self.vocab[-1].event_to_token[f'Tempo_{current_tempo}'])
                        tokens.append(bar_token)
                    current_bar += nb_new_bars
                # Position
                pos_token = [self.vocab[0].event_to_token[f'Position_{current_pos}'],
                             self.vocab[3].event_to_token[f'Position_{current_pos}'],
                             self.vocab[4].event_to_token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    pos_token.append(self.vocab[-1].event_to_token[f'Tempo_{current_tempo}'])
                tokens.append(pos_token)
            # Program (track)
            if note_event[0].desc != current_track:
                current_track = note_event[0].desc
                track_token = [self.vocab[0].event_to_token[f'Program_{current_track}'],
                               self.vocab[3].event_to_token[f'Position_{current_pos}'],
                               self.vocab[4].event_to_token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    track_token.append(self.vocab[-1].event_to_token[f'Tempo_{current_tempo}'])
                tokens.append(track_token)

            # Adding bar and position tokens to notes for positional encoding
            note_event[0] = self.vocab[0].event_to_token[f'{note_event[0].type}_{note_event[0].value}']
            note_event += [self.vocab[3].event_to_token[f'Position_{current_pos}'],
                           self.vocab[4].event_to_token[f'Bar_{current_bar}']]
            if self.additional_tokens['Tempo']:
                note_event.append(self.vocab[-1].event_to_token[f'Tempo_{current_tempo}'])
            tokens.append(note_event)

        return tokens

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, int]]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        For each note, it creates a time step as a list of tokens where:
            (list index: token type)
            0: Pitch (as an Event object for sorting purpose afterwards)
            1: Velocity
            2: Duration

        :param track: track object to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self.durations_ticks[self.current_midi_metadata['time_division']]

        tokens = []
        for note in track.notes:
            # Note
            duration = note.end - note.start
            dur_idx = np.argmin(np.abs(dur_bins - duration))
            if not track.is_drum:
                tokens.append([Event(type_='Pitch', value=note.pitch, time=note.start, desc=track.program),
                               self.vocab[1].event_to_token[f'Velocity_{note.velocity}'],
                               self.vocab[2].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_idx]))}']])
            else:
                tokens.append([Event(type_='DrumPitch', value=note.pitch, time=note.start, desc=-1),
                               self.vocab[1].event_to_token[f'Velocity_{note.velocity}'],
                               self.vocab[2].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_idx]))}']])

        # Adds chord tokens if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            chords = detect_chords(track.notes, self.current_midi_metadata['time_division'], self._first_beat_res)
            unsqueezed = []
            for c in range(len(chords)):
                chords[c].desc = track.program
                unsqueezed.append([chords[c]])
            tokens = unsqueezed + tokens  # chords at the beginning to keep the good order during sorting

        return tokens

    def tokens_to_midi(self, tokens: List[List[int]], _=None, output_path: Optional[str] = None,
                       time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        r"""Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch / Position / Bar
            (1: Velocity)
            (2: Duration)

        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param _: unused, to match parent method signature
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        midi = MidiFile(ticks_per_beat=time_division)
        midi.tempo_changes.append(TempoChange(TEMPO, 0))
        ticks_per_sample = time_division // max(self.beat_res.values())

        tracks = {}
        current_tick = 0
        current_bar = -1
        current_track = 0  # default set to piano
        for time_step in tokens:
            tok_type, tok_val = self.vocab[0].token_to_event[time_step[0]].split('_')
            if tok_type == 'Bar':
                current_bar += 1
                current_tick = current_bar * time_division * 4
            elif tok_type == 'Position':
                if current_bar == -1:
                    current_bar = 0  # as this Position token occurs before any Bar token
                current_tick = current_bar * time_division * 4 + int(tok_val) * ticks_per_sample
            elif tok_type == 'Program':
                current_track = tok_val
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif tok_type == 'Pitch' or tok_type == 'DrumPitch':
                vel, duration = (self.vocab[i].token_to_event[time_step[i]].split('_')[1] for i in (1, 2))
                if any(val == 'None' for val in (vel, duration)):
                    continue
                pitch = int(tok_val)
                vel = int(vel)
                duration = self._token_duration_to_ticks(duration, time_division)

                tracks[current_track].append(Note(vel, pitch, current_tick, current_tick + duration))

        # Appends created notes to MIDI object
        for program, notes in tracks.items():
            if int(program) == -1:
                midi.instruments.append(Instrument(0, True, 'Drums'))
            else:
                midi.instruments.append(Instrument(int(program), False, MIDI_INSTRUMENTS[int(program)]['name']))
            midi.instruments[-1].notes = notes

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def tokens_to_track(self, tokens: List[List[int]], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)):
        r"""NOT RELEVANT / IMPLEMENTED IN MUMIDI
        Use tokens_to_midi instead

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        raise NotImplementedError('tokens_to_track not implemented for Octuple, use tokens_to_midi instead')

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> List[Vocabulary]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is used as a padding index for training.
        0: Pitch / Position / Bar / Program / (Chord)
        (1: Velocity)
        (2: Duration)
        1 or 3: Current Bar embedding
        2 or 4: Current Position embedding
        (-1: Tempo)

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print('\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, '
                  '_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m')
        vocab = [Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask) for _ in range(5)]

        # PITCH
        vocab[0].add_event(f'Pitch_{i}' for i in self.pitch_range)

        # DRUM PITCHES
        vocab[0].add_event(f'DrumPitch_{i}' for i in self.drum_pitch_range)

        # VELOCITY
        vocab[1].add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        vocab[2].add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/* time signature
        vocab[0].add_event(f'Position_{i}' for i in range(nb_positions))
        vocab[3].add_event('Position_Ignore')  # special embedding for 'Bar_None' tokens
        vocab[3].add_event(f'Position_{i}' for i in range(nb_positions))  # pos enc

        # CHORD
        if self.additional_tokens['Chord']:
            vocab[0].add_event(f'Chord_{i}' for i in range(3, 6))  # non recognized chords (between 3 and 5 notes only)
            vocab[0].add_event(f'Chord_{chord_quality}' for chord_quality in CHORD_MAPS)

        # REST
        if self.additional_tokens['Rest']:
            vocab[0].add_event(f'Rest_{".".join(map(str, rest))}' for rest in self.rests)

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.append(Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask))
            vocab[-1].add_event(f'Tempo_{i}' for i in self.tempos)

        # PROGRAM
        vocab[0].add_event(f'Program_{program}' for program in self.programs)

        # BAR POS ENC
        vocab[0].add_event('Bar_None')  # new bar token
        vocab[4].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Here the combination of Pitch, Velocity and Duration tokens is represented by
        "Pitch" in the graph.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['Bar'] = ['Bar', 'Position']
        dic['Position'] = ['Program']
        dic['Program'] = ['Pitch', 'DrumPitch']
        dic['Pitch'] = ['Pitch', 'Program', 'Bar', 'Position']
        dic['DrumPitch'] = ['DrumPitch', 'Program', 'Bar', 'Position']

        if self.additional_tokens['Chord']:
            dic['Program'] += ['Chord']
            dic['Chord'] = ['Pitch']

        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(self, tokens: List[List[int]], consider_pad: bool = False) -> float:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err = 0
        previous_type = self.vocab[0].token_type(tokens[0][0])
        current_pitches = []
        bar_idx = -1 if not self.additional_tokens['Tempo'] else -2
        pos_idx = -2 if not self.additional_tokens['Tempo'] else -3
        current_bar = int(self.vocab[4].token_to_event[tokens[0][bar_idx]].split('_')[1])
        current_pos = self.vocab[3].token_to_event[tokens[0][pos_idx]].split('_')[1]
        current_pos = int(current_pos) if current_pos != 'Ignore' else -1

        for token in tokens[1:]:
            if not consider_pad and previous_type == 'PAD':
                break
            bar_value = int(self.vocab[4].token_to_event[token[bar_idx]].split('_')[1])
            pos_value = self.vocab[3].token_to_event[token[pos_idx]].split('_')[1]
            pos_value = int(pos_value) if pos_value != 'Ignore' else -1
            token_type, token_value = self.vocab[0].token_to_event[token[0]].split('_')

            if any(self.vocab[i + (2 if token_type != 'Pitch' else 0)][tok].split('_')[0]
                   in ['PAD', 'MASK'] for i, tok in enumerate(token)):
                err += 1
                continue

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == 'Bar':  # reset
                    current_bar += 1
                    current_pos = -1
                    current_pitches = []
                elif token_type == 'Pitch':
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                        continue
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == 'Position':
                    if int(token_value) <= current_pos or int(token_value) != pos_value:
                        err += 1  # token position value <= to the current position
                        continue
                    else:
                        current_pos = int(token_value)
                        current_pitches = []

                if pos_value < current_pos or bar_value < current_bar:
                    err += 1
            # Bad token type
            else:
                err += 1

            previous_type = token_type
        return err / len(tokens)
