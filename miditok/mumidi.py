""" MuMIDI encoding method, as introduced in PopMag
https://arxiv.org/abs/2008.07703

"""

from math import ceil
import json
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer, Vocabulary, Event, detect_chords, remove_duplicated_notes
from .constants import *


# recommended range from the GM2 specs
# note: the "Applause" at pitch 88 of the orchestra drum set is ignored, increase to 89 if you need it
DRUM_PITCH_RANGE = range(27, 88)


class MuMIDI(MIDITokenizer):
    """ MuMIDI encoding method, as introduced in PopMag
    https://arxiv.org/abs/2008.07703

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo)
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    :param drum_pitch_range: range of used MIDI pitches for drums exclusively
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 sos_eos_tokens: bool = False, params=None, drum_pitch_range: range = DRUM_PITCH_RANGE):
        additional_tokens['Rest'] = False
        self.drum_pitch_range = drum_pitch_range
        # used in place of positional encoding
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, params)

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
                       'drum_pitch_range': (self.drum_pitch_range.start, self.drum_pitch_range.stop),
                       'beat_res': {f'{k1}_{k2}': v for (k1, k2), v in self.beat_res.items()},
                       'nb_velocities': len(self.velocities),
                       'additional_tokens': self.additional_tokens,
                       'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding},
                      outfile)

    def midi_to_tokens(self, midi: MidiFile) -> List[List[int]]:
        """ Override the parent class method
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
        t = 0
        while t < len(midi.instruments):
            self.quantize_notes(midi.instruments[t].notes, midi.ticks_per_beat,
                                self.pitch_range if not midi.instruments[t].is_drum else self.drum_pitch_range)
            midi.instruments[t].notes.sort(key=lambda x: (x.start, x.pitch, x.end))  # sort notes
            remove_duplicated_notes(midi.instruments[t].notes)  # remove possible duplicated notes
            if len(midi.instruments[t].notes) == 0:
                del midi.instruments[t]
                continue
            t += 1
        if self.additional_tokens['Tempo']:
            self.quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        # Register MIDI metadata
        self.current_midi_metadata = {'time_division': midi.ticks_per_beat,
                                      'tempo_changes': midi.tempo_changes,
                                      'time_sig_changes': midi.time_signature_changes,
                                      'key_sig_changes': midi.key_signature_changes}

        # Check bar embedding limit, update if needed
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.max_bar_embedding < nb_bars:
            self.vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, nb_bars))
            self.max_bar_embedding = nb_bars

        # Convert each track to tokens
        note_tokens = []
        for track in midi.instruments:
            note_tokens += self.track_to_tokens(track)

        note_tokens.sort(key=lambda x: (x[0].time, x[0].desc))  # Sort by time then track

        ticks_per_sample = midi.ticks_per_beat / max(self.beat_res.values())
        ticks_per_bar = midi.ticks_per_beat * 4
        tokens = []

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track = -2  # because -2 doesnt exist
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
            # Positions and bars
            if note_event[0].time != current_tick:
                pos_index = int((note_event[0].time % ticks_per_bar) / ticks_per_sample)
                current_tick = note_event[0].time
                current_pos = pos_index
                current_track = -2  # reset
                # (New bar)
                if current_bar < current_tick // ticks_per_bar:
                    nb_new_bars = current_tick // ticks_per_bar - current_bar
                    for i in range(nb_new_bars):
                        bar_token = [self.vocab.event_to_token['Bar_None'],
                                     self.vocab.event_to_token['Position_Ignore'],
                                     self.vocab.event_to_token[f'Bar_{current_bar + i + 1}']]
                        if self.additional_tokens['Tempo']:
                            bar_token.append(self.vocab.event_to_token[f'Tempo_{current_tempo}'])
                        tokens.append(bar_token)
                    current_bar += nb_new_bars
                # Position
                pos_token = [self.vocab.event_to_token[f'Position_{current_pos}'],
                             self.vocab.event_to_token[f'Position_{current_pos}'],
                             self.vocab.event_to_token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    pos_token.append(self.vocab.event_to_token[f'Tempo_{current_tempo}'])
                tokens.append(pos_token)
            # Program (track)
            if note_event[0].desc != current_track:
                current_track = note_event[0].desc
                track_token = [self.vocab.event_to_token[f'Program_{current_track}'],
                               self.vocab.event_to_token[f'Position_{current_pos}'],
                               self.vocab.event_to_token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    track_token.append(self.vocab.event_to_token[f'Tempo_{current_tempo}'])
                tokens.append(track_token)

            # Adding bar and position tokens to notes for positional encoding
            note_event[0] = self.vocab.event_to_token[f'{note_event[0].type}_{note_event[0].value}']
            note_event += [self.vocab.event_to_token[f'Position_{current_pos}'],
                           self.vocab.event_to_token[f'Bar_{current_bar}']]
            if self.additional_tokens['Tempo']:
                note_event.append(self.vocab.event_to_token[f'Tempo_{current_tempo}'])
            tokens.append(note_event)

        return tokens

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, int]]]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        For each note, it create a time step as a list of tokens where:
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
            dur_index = np.argmin(np.abs(dur_bins - duration))
            if not track.is_drum:
                tokens.append([Event(type_='Pitch', time=note.start, value=note.pitch, desc=track.program),
                               self.vocab.event_to_token[f'Velocity_{note.velocity}'],
                               self.vocab.event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']])
            else:
                tokens.append([Event(type_='DrumPitch', time=note.start, value=note.pitch, desc=-1),
                               self.vocab.event_to_token[f'Velocity_{note.velocity}'],
                               self.vocab.event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']])

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
        """ Override the parent class method
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
        current_track = -2
        for time_step in tokens:
            events = self.tokens_to_events(time_step)
            if events[0].type == 'Bar':
                current_bar += 1
                current_tick = current_bar * time_division * 4
            elif events[0].type == 'Position':
                current_tick = current_bar * time_division * 4 + int(events[1].value) * ticks_per_sample
            elif events[0].type == 'Program':
                current_track = events[0].value
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif events[0].type == 'Pitch' or events[0].type == 'DrumPitch':
                pitch = int(events[0].value)
                vel = int(events[1].value)
                duration = self._token_duration_to_ticks(events[2].value, time_division)

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
        """ NOT RELEVANT / IMPLEMENTED IN MUMIDI
        Use tokens_to_midi instead

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        raise NotImplementedError('tokens_to_track not implemented for Octuple, use tokens_to_midi instead')

    def _create_vocabulary(self, sos_eos_tokens: bool = False) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        vocab = Vocabulary({'PAD_None': 0})

        # PITCH
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)

        # DRUM PITCHES
        vocab.add_event(f'DrumPitch_{i}' for i in self.drum_pitch_range)

        # VELOCITY
        vocab.add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        vocab.add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab.add_event('Position_Ignore')  # special embedding for 'Bar_None' tokens
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
        vocab.add_event(f'Program_{program}' for program in range(-1, 128))

        # SOS & EOS
        if sos_eos_tokens:
            vocab.add_sos_eos_to_vocab()

        # BAR --- MUST BE LAST IN DIC AS THIS MIGHT BE INCREASED
        vocab.add_event('Bar_None')  # new bar token
        vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        """ Returns a graph (as a dictionary) of the possible token
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

        self._add_pad_type_to_graph(dic)
        return dic

    def token_types_errors(self, tokens: List[List[int]], consider_pad: bool = False) -> float:
        """ Checks if a sequence of tokens is constituted of good token types
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
        previous_type = self.vocab.token_type(tokens[0][0])
        current_pitches = []
        bar_idx = -1 if not self.additional_tokens['Tempo'] else -2
        pos_idx = -2 if not self.additional_tokens['Tempo'] else -3
        current_bar = int(self.vocab.token_to_event[tokens[0][bar_idx]].split('_')[1])
        current_pos = self.vocab.token_to_event[tokens[0][pos_idx]].split('_')[1]
        current_pos = int(current_pos) if current_pos != 'Ignore' else -1

        for token in tokens[1:]:
            if not consider_pad and previous_type == 'PAD':
                break
            bar_value = int(self.vocab.token_to_event[token[bar_idx]].split('_')[1])
            pos_value = self.vocab.token_to_event[token[pos_idx]].split('_')[1]
            pos_value = int(pos_value) if pos_value != 'Ignore' else -1
            token_type, token_value = self.vocab.token_to_event[token[0]].split('_')

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
