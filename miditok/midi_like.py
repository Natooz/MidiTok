""" MIDI-like encoding method similar to ???
Music Transformer:

"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange

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
        additional_tokens['Empty'] = False  # Incompatible additional tokens
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, program_tokens, params)

    def track_to_events(self, track: Instrument) -> List[Event]:
        """ Converts a track (list of Note objects) into Event objects
        (can probably be achieved faster with Mido objects)

        :param track: track object to convert
        :return: list of events
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # it should have been done in quantization function
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            if note.pitch not in self.pitch_range:  # Notes to low or to high are discarded
                continue
            # Note On
            events.append(Event(
                name='Note-On',
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
            # Note Off
            events.append(Event(
                name='Note-Off',
                time=note.end,
                value=note.pitch,
                text=note.pitch))
        # Adds tempo events if specified
        if self.additional_tokens['Tempo']:
            for tempo_change in self.current_midi_metadata['tempo_changes']:
                events.append(Event(
                    name='Tempo',
                    time=tempo_change.time,
                    value=(np.abs(self.tempo_bins - tempo_change.tempo)).argmin(),
                    text=tempo_change.tempo))

        # Sorts events in the good order
        events.sort(key=lambda x: x.time)

        # Time Shift
        current_tick = 0
        for e, event in enumerate(events[:-1].copy()):
            if event.time == current_tick:
                continue
            time_shift = event.time - current_tick
            index = np.argmin(np.abs([ticks - time_shift for ticks in
                                      self.durations_ticks[self.current_midi_metadata['time_division']]]))
            events.append(Event(
                name='Time-Shift',
                time=current_tick,
                value='.'.join(map(str, self.durations[index])),
                text=f'{time_shift} ticks'))
            current_tick = event.time

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            events += detect_chords(track.notes, self.current_midi_metadata['time_division'])

        events.sort(key=lambda x: (x.time, self._order(x)))

        return events

    def tokens_to_midi(self, tokens: List[List[int]], programs: Optional[List[Tuple[int, bool]]] = None,
                       output_path: Optional[str] = None, time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        """ Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        NOTE: for multitrack with tempo, only the tempo tokens of the first
            decoded track will be used for the MIDI

        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param programs: programs of the tracks
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        midi = MidiFile(ticks_per_beat=time_division)
        for i, track_tokens in enumerate(tokens):
            if programs is not None:
                track, tempos = self.tokens_to_track(track_tokens, time_division, programs[i])
            else:
                track, tempos = self.tokens_to_track(track_tokens, time_division)
            midi.instruments.append(track)
            if i == 0:  # only keep tempo changes of the first track
                midi.tempo_changes = tempos
                midi.tempo_changes[0].time = 0

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def events_to_track(self, events: List[Event], time_division: int, program: Optional[Tuple[int, bool]] = (0, False),
                        default_duration: int = None) -> Tuple[Instrument, List[TempoChange]]:
        """ Transform a list of Event objects into an instrument object

        :param events: list of Event objects to convert to a track
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object
        """
        max_duration = (self.durations[-1][0] + self.durations[-1][1]) * time_division

        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        if self.additional_tokens['Tempo']:
            tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below
        else:  # default
            tempo_changes = [TempoChange(TEMPO, 0)] * 2  # the first will be deleted at the end of the method

        current_tick = 0
        count = 0
        while count < len(events):
            if events[count].name == 'Note-On':
                try:
                    if events[count + 1].name == 'Velocity':
                        pitch = int(events[count].value)
                        vel = int(self.velocity_bins[int(events[count + 1].value)])

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(count+1, len(events)):
                            if events[i].name == 'Note-Off' and int(events[i].value) == pitch:
                                duration = offset_tick
                                break
                            elif events[i].name == 'Time-Shift':
                                beat, pos, res = map(int, events[i].value.split('.'))
                                offset_tick += (beat * res + pos) * time_division // res
                            if offset_tick > max_duration:  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        count += 2
                except IndexError as _:
                    count += 1
            elif events[count].name == 'Time-Shift':
                beat, pos, res = map(int, events[count].value.split('.'))
                current_tick += (beat * res + pos) * time_division // res
                count += 1
            elif events[count].name == 'Tempo':
                tempo = int(self.tempo_bins[int(events[count].value)])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
                count += 1
            else:
                count += 1
        del tempo_changes[0]
        return instrument, tempo_changes

    def create_vocabulary(self, program_tokens: bool) -> Tuple[dict, dict, dict]:
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
        event_to_token = {'PAD_None': 0}  # starting at 1, token 0 is for padding
        token_type_indices = {'Pad': [0]}  # Empty is for empty bars
        count = 1

        # NOTE ON
        token_type_indices['Note-On'] = list(range(count, count + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Note-On_{i}'] = count
            count += 1

        # NOTE OFF
        token_type_indices['Note-Off'] = list(range(count, count + len(self.pitch_range)))
        for i in self.pitch_range:
            event_to_token[f'Note-Off_{i}'] = count
            count += 1

        # VELOCITY
        token_type_indices['Velocity'] = list(range(count, count + len(self.velocity_bins)))
        for i in range(len(self.velocity_bins)):
            event_to_token[f'Velocity_{i}'] = count
            count += 1

        # TIME SHIFTS
        token_type_indices['Time-Shift'] = list(range(count, count + len(self.durations)))
        for i in range(0, len(self.durations)):
            event_to_token[f'Time-Shift_{".".join(map(str, self.durations[i]))}'] = count
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
            dic['Tempo'] = ['Time-Shift', 'Note-On']
            if self.additional_tokens['Chord']:
                dic['Tempo'] += ['Chord']

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.name == "Program":
            return 0
        elif x.name == "Note-Off":
            return 1
        elif x.name == 'Tempo':
            return 2
        elif x.name == "Chord":
            return 3
        elif x.name == "Time-Shift":
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
