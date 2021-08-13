""" MIDI encoding base class and methods
TODO MuMIDI ? https://arxiv.org/abs/2008.07703
TODO Octotuple ? : https://arxiv.org/abs/2106.05630
TODO Tempo messages ?
TODO Control change messages (sustain, modulation, pitch bend)

"""

from sys import stdout
from pathlib import Path, PurePath
from copy import deepcopy
import json
from typing import List, Tuple, Dict, Union, Callable, Optional

import numpy as np
from miditoolkit import MidiFile, Instrument, Note

from .constants import TIME_DIVISION, CHORD_MAPS


class Event:
    """ Event class, representing a token and its characteristics"""

    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f'Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})'


class MIDITokenizer:
    """ MIDI encoding base class, containing common parameters to all encodings
    and common methods.

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
    def __init__(self, pitch_range: range, beat_res: Dict[Tuple[int, int], int], nb_velocities: int,
                 additional_tokens: Dict[str, bool], program_tokens: bool):
        self.pitch_range = pitch_range
        self.beat_res = beat_res
        self.additional_tokens = deepcopy(additional_tokens)

        self.durations = self.create_durations_tuples()
        self.velocity_bins = np.linspace(0, 127, nb_velocities + 1, dtype=np.intc)
        np.delete(self.velocity_bins, 0)  # removes velocity 0
        self.event2token, self.token2event, self.token_types_indices = self.create_token_dicts(program_tokens)

        # keep in memory durations in ticks for seen time divisions so these values
        # are not calculated each time a MIDI is processed
        self.durations_ticks = {}

    def midi_to_tokens(self, midi: MidiFile) -> Tuple[List[List[int]], List[Tuple[int, bool]]]:
        """ Converts a MIDI file in a tokens representation

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

        tokens = []
        for track in midi.instruments:
            tokens.append(self.track_to_tokens(track, midi.ticks_per_beat))

        track_info = [(int(track.program), track.is_drum) for track in midi.instruments]
        return tokens, track_info

    def track_to_tokens(self, track: Instrument, time_division: int) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :return: sequence of corresponding tokens
        """
        quantize_note_times(track.notes, time_division, max(self.beat_res.values()))  # adjusts notes timings
        track.notes.sort(key=lambda x: (x.start, x.pitch))  # sort notes
        remove_duplicated_notes(track.notes)  # remove possible duplicated notes
        events = self.track_to_events(track.notes, time_division, track.is_drum)  # get distinct events
        return self.events_to_tokens(events)

    def events_to_tokens(self, events: List[Event]) -> List[int]:
        """ Converts a list of Event objects into a list of tokens
        You can override this method if necessary (e.g. CP Word)

        :param events: list of Events objects to convert
        :return: list of corresponding tokens
        """
        return [self.event2token[f'{event.name}_{event.value}'] for event in events]

    def tokens_to_events(self, tokens: List[int]) -> List[Event]:
        """ Convert a sequence of tokens in their respective event objects
        You can override this method if necessary (e.g. CP Word)

        :param tokens: sequence of tokens to convert
        :return: the sequence of corresponding events
        """
        events = []
        for token in tokens:
            name, val = self.token2event[token].split('_')
            events.append(Event(name, None, val, None))
        return events

    def track_to_events(self, notes: List[Note], time_division: int, drum: Optional[bool] = False) -> List[Event]:
        """ Converts a track (list of Note objects) into Event objects
        NOTE: this method must take care of chord or other types of tokens, if specified
        And to sort every events in the right order!

        :param notes: notes of the track to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param drum: specify if the notes treated are from a drum track (if it is the case no chord should be detected)
        :return: list of events
                 the events should be in the order Bar -> Position -> Chord -> Pitch -> Velocity -> Duration
        """
        raise NotImplementedError

    def tokens_to_midi(self, tokens: List[List[int]], programs: Optional[List[Tuple[int, bool]]] = None,
                       output_path: Optional[str] = None, time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        """ Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.

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
                midi.instruments.append(self.tokens_to_track(track_tokens, time_division, programs[i]))
            else:
                midi.instruments.append(self.tokens_to_track(track_tokens, time_division))

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Instrument:
        """ Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return:
        """
        events = self.tokens_to_events(tokens)
        return self.events_to_track(events, time_division, program)

    def events_to_track(self, events: List[Event], time_division: int, program: Optional[int] = 0) -> Instrument:
        """ Transform a list of Event objects into an instrument object

        :param events: list of Event objects to convert to a track
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track, use -1 for drums (default 0, piano)
        :return: the miditoolkit instrument object
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def create_token_types_graph(self) -> Dict[str, List[str]]:
        """ Creates a dictionary for the directions of the token types of the encoding"""
        raise NotImplementedError

    def create_durations_tuples(self) -> List[Tuple]:
        """ Creates the possible durations in bar / beat units, as tuple of the form:
        (beat, pos, res) where beat is the number of beats, pos the number of "frames"
        ans res the beat resolution considered (frames per beat)
        Example: (2, 5, 8) means the duration is 2 beat long + position 5 / 8 of the ongoing beat
        In pure ticks we have: duration = (beat * res + pos) * time_division // res
            Is equivalent to: duration = nb_of_frames * ticks_per_frame
        So in the last example, if time_division is 384: duration = (2 * 8 + 5) * 384 // 8 = 1008 ticks

        :return: the duration bins
        """
        durations = []
        for beat_range, beat_res in self.beat_res.items():
            durations += [(beat, pos, beat_res) for beat in range(*beat_range) for pos in range(beat_res)]
        durations += [(max(max(self.beat_res)), 0, self.beat_res[max(self.beat_res)])]  # the last one
        del durations[0]  # removes duration of 0
        return durations

    def tokenize_midi_dataset(self, midi_paths: Union[List[str], List[Path], List[PurePath]],
                              out_dir: Union[str, Path, PurePath], validation_fn: Callable[[MidiFile], bool] = None,
                              logging: bool = True):
        """ Converts a dataset / list of MIDI files, into their token version and save them as json files

        :param midi_paths: paths of the MIDI files
        :param out_dir: output directory to save the converted files
        :param validation_fn: a function checking if the MIDI is valid on your requirements
                            (e.g. time signature, minimum/maximum length, instruments ...)
        :param logging: logs a progress bar
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.save_params(out_dir)  # Saves the parameters with which the MIDIs are converted

        for m, midi_path in enumerate(midi_paths):
            if logging:
                bar_len = 60
                filled_len = int(round(bar_len * m / len(midi_paths)))
                percents = round(100.0 * m / len(midi_paths), 2)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                prog = f'\r{m} / {len(midi_paths)} [{bar}] {percents:.1f}% ...Converting MIDIs to tokens: {midi_path}'
                stdout.write(prog)
                stdout.flush()

            # Some MIDIs can contains errors that are raised by Mido, if so the loop continues
            try:
                midi = MidiFile(PurePath(midi_path))
            except Exception as _:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # Passing the MIDI to validation tests if given
            if validation_fn is not None:
                if not validation_fn(midi):
                    continue

            # Converting the MIDI to tokens and saving them as json
            tokens, track_info = self.midi_to_tokens(midi)
            with open(PurePath(out_dir, midi_path).with_suffix(".json"), 'w') as outfile:
                json.dump([tokens[0], track_info[0]], outfile)

    def save_params(self, out_dir: Union[str, Path, PurePath]):
        """ Saves the base parameters of this encoding in a txt file
        Useful to keep track of how a dataset has been tokenized / encoded

        :param out_dir: output directory to save the file
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(PurePath(out_dir, 'config').with_suffix(".txt"), 'w') as outfile:
            json.dump({'pitch_range': (self.pitch_range.start, self.pitch_range.stop),
                       'beat_res': self.beat_res, 'nb_velocities': len(self.velocity_bins),
                       'additional_tokens': self.additional_tokens}, outfile)


def quantize_note_times(notes: List[Note], time_division: int, beat_res: int):
    """ Quantize the notes items start and end values.
    It shifts the notes so they start at times that match the quantization (e.g. 16 frames per bar)

    :param notes: notes to quantize
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param beat_res: number of frames (time steps, or positions) per beat
    """
    ticks = int(time_division / beat_res)
    quantized_ticks = np.arange(0, max([n.end for n in notes]) + 2 * ticks, ticks, dtype=int)
    for i, note in enumerate(notes):  # items are notes
        note.start = quantized_ticks[np.argmin(np.abs(quantized_ticks - note.start))]
        note.end = quantized_ticks[np.argmin(np.abs(quantized_ticks - note.end))]

        if note.start == note.end:  # if this happens to often, consider using a higher beat resolution
            note.end += ticks  # like 8 frames per beat or 24 frames per bar


def remove_duplicated_notes(notes: List[Note]):
    """ Remove possible duplicated notes, i.e. with the same pitch, starting and ending times.
    Before running this function make sure the notes has been sorted by start and pitch:
    notes.sort(key=lambda x: (x.start, x.pitch))

    :param notes: notes to analyse
    """
    for i in range(len(notes) - 1, 0, -1):  # removing possible duplicated notes
        if notes[i].pitch == notes[i - 1].pitch and notes[i].start == notes[i - 1].start and \
                notes[i].end == notes[i - 1].end:
            del notes[i]


def detect_chords(notes: List[Note], time_division: int) -> List[Event]:
    """ Chord detection method.
    NOTE: on very large tracks with high note density this method can be very slow !
    If you plan to use it with the Maestro or GiantMIDI datasets, it can take up to
    hundreds of seconds per MIDI depending on your cpu.
    One time step at a time, it will analyse the notes played together
    and detect possible chords.

    :param notes: notes to analyse
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :return: the detected chords as Event objects
    """
    tuples = []
    for note in notes:
        tuples.append((note.pitch, int(note.start), int(note.end)))
    notes = np.asarray(tuples)

    count = 0
    chords = []
    while count < len(notes):
        # Gather the notes around the same time step
        onset_notes = notes[count:]
        onset_notes = onset_notes[np.where(onset_notes[:, 1] <= notes[count][1] + time_division / 4)]

        # If it is ambiguous, e.g. the notes lengths are too different
        if np.any(np.abs(onset_notes[:, 2] - onset_notes[0, 2]) > time_division / 2):
            count += len(onset_notes)
            continue

        # Selects the possible chords notes
        if notes[count][2] - notes[count][1] <= time_division / 2:
            onset_notes = onset_notes[np.where(onset_notes[:, 1] == onset_notes[0][1])]
        chord = onset_notes[np.where(onset_notes[:, 2] - onset_notes[0, 2] <= time_division / 2)]

        # Creates the "chord map" and see if it has a "known" quality, append a chord event if it is valid
        chord_map = (chord[:, 0] - chord[0, 0]).tolist()
        if 3 <= len(chord_map) <= 5 and chord_map[-1] <= 24:  # max interval between the root and highest degree
            chord_quality = len(chord)
            for quality, known_chord in CHORD_MAPS.items():
                if known_chord == chord_map:
                    chord_quality = quality
                    break
            chords.append((chord_quality, min(chord[:, 1]), chord_map))
        count += len(onset_notes)  # Move to the next notes

    events = []
    for chord in chords:
        events.append(Event('Chord', chord[1], chord[0], chord[2]))
    return events


def _detect_chords_python(notes: List[Note], time_division: int) -> List[Event]:
    """ DEPRECIATED
    Old chord detection method, equivalent to detect_chords but 100% python
    The code is more elegant but slower

    :param notes: notes to analyse
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :return: the detected chords as Event objects
    """
    count = 0
    chords = []
    while count < len(notes):
        # Gather the notes around the same time step
        # onset_notes = [note for note in notes if abs(note.start - notes[count].start) <= time_division / 4]
        onset_notes = [note for note in notes[count:] if note.start - notes[count].start <= time_division / 4]

        # If it is ambiguous, e.g. the notes lengths are too different
        if any(abs(note.end - notes[count].end) > time_division / 2 for note in onset_notes):
            count += len(onset_notes)
            continue

        # Selects the possible chords notes
        if notes[count].end - notes[count].start <= time_division / 2:
            onset_notes = [note for note in notes[count:] if note.start == notes[count].start]
        chord = [note for note in onset_notes if abs(note.end - notes[count].end) <= time_division / 2]

        # Creates the "chord map" and see if it has a "known" quality, append a chord event if it is valid
        chord_map = [chord[i].pitch - chord[0].pitch for i in range(len(chord))]
        if 3 <= len(chord_map) <= 5 and chord_map[-1] <= 24:  # max interval between the root and highest degree
            if chord_map in CHORD_MAPS.values():
                chord_quality = list(CHORD_MAPS.keys())[list(CHORD_MAPS.values()).index(chord_map)]
                chords.append(Event('Chord', min(note.start for note in chord), chord_quality, chord_map))
            else:
                chords.append(Event('Chord', min(note.start for note in chord), len(chord), chord_map))
        count += len(onset_notes)  # Move to the next notes
    return chords


def merge_tracks(tracks: List[Instrument]) -> Instrument:
    """ Merge several miditoolkit Instrument objects
    It will take the first object, and concat the notes of the others

    :param tracks: list of tracks to merge
    :return: the merged track
    """
    tracks[0].name += ''.join([' / ' + t.name for t in tracks[1:]])
    tracks[0].notes = sum((t.notes for t in tracks), [])
    tracks[0].notes.sort(key=lambda note: note.start)
    return tracks[0]


def current_bar_pos(seq: List[int], bar_token: int, position_tokens: List[int], pitch_tokens: List[int],
                    chord_tokens: List[int] = None) -> Tuple[int, int, List[int], bool]:
    """ Detects the current state of a sequence of tokens

    :param seq: sequence of tokens
    :param bar_token: the bar token value
    :param position_tokens: position tokens values
    :param pitch_tokens: pitch tokens values
    :param chord_tokens: chord tokens values
    :return: the current bar, current position within the bar, current pitches played at this position,
            and if a chord token has been predicted at this position
    """
    # Current bar
    bar_idx = [i for i, token in enumerate(seq) if token == bar_token]
    current_bar = len(bar_idx)
    # Current position value within the bar
    pos_idx = [i for i, token in enumerate(seq[bar_idx[-1]:]) if token in position_tokens]
    current_pos = len(pos_idx) - 1  # position value, e.g. from 0 to 15, -1 means a bar with no Pos token following
    # Pitches played at the current position
    current_pitches = [token for token in seq[pos_idx[-1]:] if token in pitch_tokens]
    # Chord predicted
    if chord_tokens is not None:
        chord_at_this_pos = any(token in chord_tokens for token in seq[pos_idx[-1]:])
    else:
        chord_at_this_pos = False
    return current_bar, current_pos, current_pitches, chord_at_this_pos
