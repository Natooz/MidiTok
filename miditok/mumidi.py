""" MuMIDI encoding method, as introduced in PopMag
https://arxiv.org/abs/2008.07703

"""

from math import ceil
import json
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note

from .midi_tokenizer_base import MIDITokenizer, Event, detect_chords, quantize_note_times, quantize_tempos, \
    remove_duplicated_notes
from .constants import *


# recommended range from the GM2 specs
# note: the "Applause" at pitch 88 of the orchestra drum set is ignored, increase to 89 if you need it
DRUM_PITCH_RANGE = range(27, 88)


class MuMIDIEncoding(MIDITokenizer):
    """ MuMIDI encoding method, as introduced in PopMag
    https://arxiv.org/abs/2008.07703

    :param pitch_range: range of used MIDI pitches
    :param drum_pitch_range: range of used MIDI pitches for drums exclusively
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
    def __init__(self, pitch_range: range = PITCH_RANGE, drum_pitch_range: range = DRUM_PITCH_RANGE,
                 beat_res: Dict[Tuple[int, int], int] = BEAT_RES, nb_velocities: int = NB_VELOCITIES,
                 additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS, program_tokens: bool = PROGRAM_TOKENS,
                 params=None):
        self.drum_pitch_range = drum_pitch_range
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
                       'drum_pitch_range': (self.drum_pitch_range.start, self.drum_pitch_range.stop),
                       'beat_res': self.beat_res, 'nb_velocities': len(self.velocity_bins),
                       'additional_tokens': self.additional_tokens, 'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding},
                      outfile)

    def midi_to_tokens(self, midi: MidiFile) -> List[List[int]]:
        """ Override the parent class method
        Converts a MIDI file in a tokens representation

        :param midi: the MIDI objet to convert
        :return: the token representation :
                  1. tracks converted into sequences of tokens
                  2. program numbers and if it is drums, for each track
        """
        try:
            _ = self.durations_ticks[midi.ticks_per_beat]
        except KeyError:
            self.durations_ticks[midi.ticks_per_beat] = [(beat * res + pos) * midi.ticks_per_beat // res
                                                         for beat, pos, res in self.durations]

        self.current_midi_metadata = {'time_division': midi.ticks_per_beat,
                                      'tempo_changes': midi.tempo_changes,
                                      'time_sig_changes': midi.time_signature_changes,
                                      'key_sig_changes': midi.key_signature_changes}
        quantize_tempos(midi.tempo_changes, midi.ticks_per_beat, max(self.beat_res.values()))

        ticks_per_frame = midi.ticks_per_beat // max(self.beat_res.values())
        ticks_per_bar = midi.ticks_per_beat * 4

        # Process notes of every tracks
        note_events = []
        for track in midi.instruments:
            quantize_note_times(track.notes, midi.ticks_per_beat, max(self.beat_res.values()))  # adjusts notes timings
            track.notes.sort(key=lambda x: (x.start, x.pitch))  # sort notes
            remove_duplicated_notes(track.notes)  # remove possible duplicated notes
            note_events += self.track_to_events(track)

            # Check bar embedding limit, update if needed
            nb_bars = ceil(max(note.end for note in track.notes) / ticks_per_bar)
            if self.max_bar_embedding < nb_bars:
                count = len(self.event2token)
                for i in range(self.max_bar_embedding, nb_bars):
                    self.event2token[f'Bar_{i}'] = count
                    self.token2event[count] = f'Bar_{i}'
                    self.token_types_indices['Bar'] += [count]
                    count += 1
                self.max_bar_embedding = nb_bars

        note_events.sort(key=lambda x: (x[0].time, x[0].text, x[0].value))  # Sort by time then track then pitch

        tokens = []

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track = -2  # doesnt exist
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        current_tempo = (np.abs(self.tempo_bins - current_tempo)).argmin()
        for note_event in note_events:
            # (Tempo) update tempo values current_tempo
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
            # Positions and bars
            if note_event[0].time != current_tick:
                pos_index = int((note_event[0].time % ticks_per_bar) / ticks_per_frame)
                current_tick = note_event[0].time
                current_pos = pos_index
                current_track = -2  # reset
                # (New bar)
                if current_bar < current_tick // ticks_per_bar:
                    nb_new_bars = current_tick // ticks_per_bar - current_bar
                    for i in range(nb_new_bars):
                        bar_token = [self.event2token['Bar_None'],
                                     self.event2token['Position_Ignore'],
                                     self.event2token[f'Bar_{current_bar + i + 1}']]
                        if self.additional_tokens['Tempo']:
                            bar_token.append(self.event2token[f'Tempo_{current_tempo}'])
                        tokens.append(bar_token)
                        # (Empty bar)
                        if self.additional_tokens['Empty'] and i+1 != nb_new_bars and nb_new_bars > 1:
                            empty_token = [self.event2token['Empty_None'],
                                           self.event2token['Position_Ignore'],
                                           self.event2token[f'Bar_{current_bar + i + 1}']]
                            if self.additional_tokens['Tempo']:
                                empty_token.append(self.event2token[f'Tempo_{current_tempo}'])
                            tokens.append(empty_token)
                    current_bar += nb_new_bars
                # Position
                pos_token = [self.event2token['Position_None'],
                             self.event2token[f'Position_{current_pos}'],
                             self.event2token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    pos_token.append(self.event2token[f'Tempo_{current_tempo}'])
                tokens.append(pos_token)
            # Tracks (programs)
            if note_event[0].text != current_track:
                current_track = note_event[0].text
                track_token = [self.event2token[f'Program_{current_track}'],
                               self.event2token[f'Position_{current_pos}'],
                               self.event2token[f'Bar_{current_bar}']]
                if self.additional_tokens['Tempo']:
                    track_token.append(self.event2token[f'Tempo_{current_tempo}'])
                tokens.append(track_token)

            # Adding bar and position tokens to notes for positional encoding
            note_event[0] = self.event2token[f'{note_event[0].name}_{note_event[0].value}']
            note_event += [self.event2token[f'Position_{current_pos}'], self.event2token[f'Bar_{current_bar}']]
            if self.additional_tokens['Tempo']:
                note_event.append(self.event2token[f'Tempo_{current_tempo}'])
            tokens.append(note_event)

        return tokens

    def track_to_events(self, track: Instrument) -> List[List[Event]]:
        """ Converts a track (list of Note objects) into Event objects
        This only create pitch, velocity and duration events

        :param track: track object to convert
        :return: list of events
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens

        events = []
        for note in track.notes:
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            # Note
            velocity_index = (np.abs(self.velocity_bins - note.velocity)).argmin()
            duration = note.end - note.start
            dur_index = np.argmin(np.abs([ticks - duration for ticks in
                                          self.durations_ticks[self.current_midi_metadata['time_division']]]))
            if not track.is_drum:
                events.append([Event(name='Pitch', time=note.start, value=note.pitch, text=track.program),
                               self.event2token[f'Velocity_{velocity_index}'],
                               self.event2token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']])
            else:
                events.append([Event(name='DrumPitch', time=note.start, value=note.pitch, text=-1),
                               self.event2token[f'Velocity_{velocity_index}'],
                               self.event2token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']])

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            chords = detect_chords(track.notes, self.current_midi_metadata['time_division'])
            unsqueezed = []
            for c in range(len(chords)):
                chords[c].text = track.program
                unsqueezed.append([chords[c]])
            events = unsqueezed + events  # chords at the beginning to keep the good order during sorting

        return events

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
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_frame = time_division // max(self.beat_res.values())

        tracks = {}
        current_tick = 0
        current_bar = -1
        current_track = -2
        for time_step in tokens:
            events = self.tokens_to_events(time_step)
            if events[0].name == 'Bar':
                current_bar += 1
                current_tick = current_bar * time_division * 4
            elif events[0].name == 'Position':
                current_tick = current_bar * time_division * 4 + int(events[1].value) * ticks_per_frame
            elif events[0].name == 'Program':
                current_track = events[0].value
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif events[0].name == 'Pitch' or events[0].name == 'DrumPitch':
                pitch = int(events[0].value)
                vel = int(self.velocity_bins[int(events[1].value)])
                beat, pos, res = map(int, events[2].value.split('.'))
                duration = (beat * res + pos) * time_division // res

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

    def events_to_track(self, events: List[Event], time_division: int,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Instrument:
        """ NOT IMPLEMENTED, USE tokens_to_midi
        Transform a list of Event objects into an instrument object

        :param events: list of Event objects to convert to a track
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object
        """
        raise NotImplementedError('event_to_track not implemented for MuMIDI, use tokens_to_midi instead')

    def create_vocabulary(self, _) -> Tuple[dict, dict, dict]:
        """ Create the tokens <-> event dictionaries
        These dictionaries are created arbitrary according to constants defined
        at the top of this file.
        Note that when using them (prepare_data method), there is no error-handling
        so you must be sure that every case is covered by the dictionaries.
        NOTE: token index 0 is often used as a padding index during training, it might
        be preferable to leave it as it to pad your batch sequences
        NOTE 2: with MuMIDI track tokens are part of the representation so
        included in the vocabulary, called "Program" as for other encodings

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

        # DRUM PITCHES
        token_type_indices['DrumPitch'] = list(range(count, count + len(self.drum_pitch_range)))
        for i in self.drum_pitch_range:
            event_to_token[f'DrumPitch_{i}'] = count
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

        # BAR
        token_type_indices['Bar'] = list(range(count, count + self.max_bar_embedding + 1))
        event_to_token['Bar_None'] = count  # new bar token
        count += 1
        for i in range(self.max_bar_embedding):  # bar embeddings (positional encoding)
            event_to_token[f'Bar_{i}'] = count
            count += 1

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        token_type_indices['Position'] = list(range(count, count + nb_positions + 2))
        event_to_token['Position_None'] = count  # new position token
        event_to_token['Position_Ignore'] = count + 1  # special embedding associated with 'Bar_None' tokens
        count += 2
        for i in range(0, nb_positions):  # position embeddings (positional encoding)
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

        # EMPTY
        if self.additional_tokens['Empty']:
            event_to_token['Empty_None'] = count
            token_type_indices['Empty'] = [count]
            count += 1

        # PROGRAM
        token_type_indices['Program'] = list(range(count, count + 129))
        for program in range(-1, 128):  # -1 is drums
            event_to_token[f'Program_{program}'] = count
            count += 1

        token_to_event = {v: k for k, v in event_to_token.items()}  # inversion
        return event_to_token, token_to_event, token_type_indices

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        dic = dict()

        dic['Bar'] = ['Bar', 'Position']
        dic['Position'] = ['Program']
        dic['Program'] = ['PitchVelDur']
        dic['PitchVelDur'] = ['PitchVelDur', 'Program', 'Bar', 'Position']

        if self.additional_tokens['Chord']:
            dic['Position'] += ['Chord']
            dic['Chord'] += ['Program']

        return dic
