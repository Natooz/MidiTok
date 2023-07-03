from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..classes import Event, TokSequence, TokenizerConfig
from ..constants import (
    MIDI_INSTRUMENTS,
    TEMPO,
    TIME_DIVISION,
    TIME_SIGNATURE,
    MMM_DENSITY_BINS_MAX,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..utils import detect_chords


class MMM(MIDITokenizer):
    r"""MMM, standing for [Multi-Track Music Machine](https://arxiv.org/abs/2008.06048), is a multitrack tokenization
    primarily designed for music inpainting and infilling.
    Tracks are tokenized independently and concatenated into a single token sequence.
    ``Bar_Fill`` tokens are used to specify the bars to fill (or inpaint, or rewrite), the new tokens are then
    autoregressively generated.
    Note that *this implementation represents note durations with ``Duration`` tokens* instead of the ``NoteOff``
    strategy of the [original paper](https://arxiv.org/abs/2008.06048). The reason being that ``NoteOff`` tokens perform
    poorer for generation with causal models.

    :param tokenizer_config: the tokenizer's configuration, as a :class:`miditok.classes.TokenizerConfig` object.
    :param density_bins_max: tuple specifying the number of density bins, and the maximum density in
            notes per beat to consider. (default: (10, 20))
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        density_bins_max: Tuple[int, int] = MMM_DENSITY_BINS_MAX,
        params: Optional[Union[str, Path]] = None,
    ):
        if (
            tokenizer_config is not None
            and "density_bins_max" not in tokenizer_config.additional_params
        ):
            tokenizer_config.additional_params["density_bins_max"] = density_bins_max
        super().__init__(tokenizer_config, True, params)

    def _tweak_config_before_creating_voc(self):
        self.config.use_programs = True
        self.config.use_rests = False
        # Recreate densities here just in case density_bins_max was loaded from params (list to np array)
        if "note_densities" in self.config.additional_params:
            if isinstance(
                self.config.additional_params["note_densities"], (list, tuple)
            ):
                self.config.additional_params["note_densities"] = np.array(
                    self.config.additional_params["note_densities"]
                )
        else:
            self.config.additional_params["note_densities"] = np.linspace(
                0,
                self.config.additional_params["density_bins_max"][1],
                self.config.additional_params["density_bins_max"][0] + 1,
                dtype=np.intc,
            )

    def track_to_tokens(self, track: Instrument) -> List[Event]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of Event (:class:`miditok.Event`).

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self._current_midi_metadata["time_division"]
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        note_density_bins = self.config.additional_params["note_densities"]

        # Creates first events
        note_density = len(track.notes) / self._current_midi_metadata["max_tick"]
        note_density = int(np.argmin(np.abs(note_density_bins - note_density)))
        events: List[Event] = [
            Event("Track", "Start", 0),
            Event(
                type="Program",
                value=-1 if track.is_drum else track.program,
                time=0,
            ),
            Event(
                type="NoteDensity",
                value=note_density,
                time=0,
            ),
            Event(
                type="Bar",
                value="Start",
                time=0,
            ),
        ]

        # (Chord)
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                self._current_midi_metadata["time_division"],
                chord_maps=self.config.chord_maps,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_nb_notes_range=self.config.chord_unknown,
            )
            for chord in chords:
                events.append(chord)

        # (Tempo)
        if self.config.use_tempos:
            for tempo_change in self._current_midi_metadata["tempo_changes"]:
                events.append(
                    Event(
                        type="Tempo",
                        value=tempo_change.tempo,
                        time=tempo_change.time,
                        desc=tempo_change.tempo,
                    )
                )

        # (Time signature)
        if self.config.use_time_signatures:
            for time_sig in self._current_midi_metadata["time_sig_changes"]:
                events.append(
                    Event(
                        type="TimeSig",
                        value=f"{time_sig.numerator}/{time_sig.denominator}",
                        time=time_sig.time,
                        desc=(time_sig.numerator, time_sig.denominator),
                    )
                )

        # Note events
        for note in track.notes:
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events += [
                Event(type="Pitch", value=note.pitch, time=note.start, desc=note.pitch),
                Event(
                    type="Velocity",
                    value=note.velocity,
                    time=note.start,
                    desc=f"{note.velocity}",
                ),
                Event(
                    type="Duration",
                    value=".".join(map(str, self.durations[index])),
                    time=note.start,
                    desc=f"{duration} ticks",
                ),
            ]

        # Time events
        events.sort(key=lambda x: (x.time, self._order(x)))
        time_sig_change = self._current_midi_metadata["time_sig_changes"][0]
        first_time_sig = self._reduce_time_signature(
            time_sig_change.numerator, time_sig_change.denominator
        )
        ticks_per_bar = time_division * first_time_sig[0]
        previous_tick = 0
        current_bar = 0
        for ei in range(len(events)):
            if events[ei].type == "TimeSig":
                ticks_per_bar = time_division * events[ei].desc[0]
            if events[ei].time != previous_tick:
                # Bar
                nb_new_bars = events[ei].time // ticks_per_bar - current_bar
                if nb_new_bars > 0:
                    for i in range(nb_new_bars):
                        events += [
                            Event(
                                type="Bar",
                                value="End",
                                time=(current_bar + i + 1) * ticks_per_bar,
                                desc=0,
                            ),
                            Event(
                                type="Bar",
                                value="Start",
                                time=(current_bar + i + 1) * ticks_per_bar,
                                desc=0,
                            ),
                        ]

                    current_bar += nb_new_bars
                    previous_tick = current_bar * ticks_per_bar

                # TimeShift
                if events[ei].time != previous_tick:
                    time_shift = events[ei].time - previous_tick
                    index = np.argmin(np.abs(dur_bins - time_shift))
                    events.append(
                        Event(
                            type="TimeShift",
                            value=".".join(map(str, self.durations[index])),
                            time=previous_tick,
                            desc=f"{time_shift} ticks",
                        )
                    )

                previous_tick = events[ei].time

        events.sort(key=lambda x: (x.time, self._order(x)))
        events += [
            Event("Bar", "End", events[-1].time + 1),
            Event("Track", "End", events[-1].time + 1),
        ]
        return events

    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> None:
        r"""NOT RELEVANT / IMPLEMENTED FOR MMM, use tokens_to_midi instead

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drums, (default (0, False), piano)
        :return: None
        """
        pass

    @_out_as_complete_seq
    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        Tokenization treating all tracks as a single token sequence might
        override this method, e.g. Octuple or PopMAG.

        :param midi: the MIDI object to convert.
        :return: sequences of tokens.
        """
        # Convert each track to tokens
        events = []
        for track in midi.instruments:
            events += self.track_to_tokens(track)

        tok_seq = TokSequence(events=events)
        self.complete_sequence(tok_seq)
        return tok_seq

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        _=None,
        output_path: Optional[str] = None,
        time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param _: unused, to match parent method signature
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        tokens = cast(TokSequence, tokens)
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
            time_division % max(self.config.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.config.beat_res.values())}"
        tokens = cast(List[str], tokens.tokens)  # for reducing type errors
        ticks_per_sample = time_division // max(self.config.beat_res.values())

        # RESULTS
        instruments: List[Instrument] = []
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below
        time_signature_changes = [
            TimeSignature(*TIME_SIGNATURE, 0)
        ]  # mock the first time signature change to optimize below
        ticks_per_bar = time_division * TIME_SIGNATURE[0]  # init

        current_tick = 0
        current_bar = -1
        previous_note_end = 0  # unused (rest)
        for ti, token in enumerate(tokens):
            tok_type, tok_val = token.split("_")
            if tok_type == "Program":
                current_program = int(tok_val)
                instruments.append(
                    Instrument(
                        program=0 if current_program == -1 else current_program,
                        is_drum=current_program == -1,
                        name="Drums"
                        if current_program == -1
                        else MIDI_INSTRUMENTS[current_program]["name"],
                    )
                )
                current_tick = 0
                current_bar = -1
                previous_note_end = 0
            elif token == "Bar_Start":
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif tok_type == "Rest":
                beat, pos = map(int, tokens[ti].split("_")[1].split("."))
                if (
                    current_tick < previous_note_end
                ):  # if in case successive rest happen
                    current_tick = previous_note_end
                current_tick += beat * time_division + pos * ticks_per_sample
                current_bar = current_tick // ticks_per_bar
            elif tok_type == "TimeShift":
                if current_bar == -1:
                    # as this Position token occurs before any Bar token
                    current_bar = 0
                current_tick += self._token_duration_to_ticks(tok_val, time_division)
            elif tok_type == "Tempo":
                # If the tokenizer includes tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                tempo = int(token.split("_")[1])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            elif tok_type == "TimeSig":
                num, den = self._parse_token_time_signature(token.split("_")[1])
                current_time_signature = time_signature_changes[-1]
                if (
                    num != current_time_signature.numerator
                    and den != current_time_signature.denominator
                ):
                    time_signature_changes.append(TimeSignature(num, den, current_tick))
            elif tok_type == "Pitch":
                try:
                    if (
                        tokens[ti + 1].split("_")[0] == "Velocity"
                        and tokens[ti + 2].split("_")[0] == "Duration"
                    ):
                        pitch = int(tokens[ti].split("_")[1])
                        vel = int(tokens[ti + 1].split("_")[1])
                        duration = self._token_duration_to_ticks(
                            tokens[ti + 2].split("_")[1], time_division
                        )
                        instruments[-1].notes.append(
                            Note(vel, pitch, current_tick, current_tick + duration)
                        )
                        previous_note_end = max(
                            previous_note_end, current_tick + duration
                        )
                except (
                    IndexError
                ):  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        if len(time_signature_changes) > 1:
            del time_signature_changes[0]  # delete mocked time signature change
        time_signature_changes[0].time = 0
        # create MidiFile
        midi.instruments = instruments
        midi.tempo_changes = tempo_changes
        midi.time_signature_changes = time_signature_changes
        midi.max_tick = max(
            [
                max([note.end for note in track.notes]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )
        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def _create_base_vocabulary(
        self, sos_eos_tokens: Optional[bool] = None
    ) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # TRACK / BAR / FILL
        vocab += ["Track_Start", "Track_End"]
        vocab += ["Bar_Start", "Bar_End", "Bar_Fill"]
        vocab += ["Fill_Start", "Fill_End"]

        # PITCH
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # TIME SHIFTS
        vocab += [
            f'TimeShift_{".".join(map(str, self.durations[i]))}'
            for i in range(len(self.durations))
        ]

        # NOTE DENSITY
        vocab += [
            f"NoteDensity_{i}" for i in self.config.additional_params["note_densities"]
        ]

        # TIME SIGNATURE
        if self.config.use_time_signatures:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # CHORD
        if self.config.use_chords:
            vocab += self._create_chords_tokens()

        # TEMPO
        if self.config.use_tempos:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # PROGRAM
        vocab += [f"Program_{program}" for program in self.config.programs]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.

        :return: the token types transitions dictionary
        """
        dic: Dict[str, List[str]] = dict()

        dic["Bar"] = ["Bar", "TimeShift", "Pitch", "Track"]
        dic["TimeShift"] = ["Pitch"]
        dic["Track"] = ["Program", "Track"]
        dic["Program"] = ["NoteDensity"]
        dic["NoteDensity"] = ["Bar"]
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Pitch", "TimeShift", "Bar"]

        if self.config.use_time_signatures:
            dic["Bar"] += ["TimeSig"]
            dic["TimeSig"] = ["Pitch", "TimeShift"]

        if self.config.use_chords:
            dic["Chord"] = ["TimeShift", "Pitch"]
            dic["Bar"] += ["Chord"]
            dic["TimeShift"] += ["Chord"]

        if self.config.use_tempos:
            dic["Tempo"] = ["TimeShift", "Pitch", "Bar"]
            dic["Bar"] += ["Tempo"]
            dic["TimeShift"] += ["Tempo"]
            if self.config.use_time_signatures:
                dic["TimeSig"] += ["Tempo"]

        dic["Fill"] = list(dic.keys())

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order.

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Track" and x.value == "Start":
            return 0
        elif x.type == "Program":
            return 1
        elif x.type == "NoteDensity":
            return 2
        # elif x.type == "Fill_Start":
        #    return 2

        elif x.type == "Bar" and x.value == "End":
            return 3
        elif x.type == "Bar" and x.value == "Start":
            return 4
        elif x.type == "TimeSig":
            return 5
        elif x.type == "Tempo":
            return 6

        elif x.type == "TimeShift":
            return 8

        # elif x.type == "Fill_End":
        #    return 7
        elif x.type == "Track" and x.value == "End":
            return 10
        else:  # for other types of events, the order should be handled when inserting the events in the sequence
            return 7

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(
        self, tokens_to_check: Union[TokSequence, List[Union[int, List[int]]]]
    ) -> float:
        tokens_to_check = cast(TokSequence, tokens_to_check)
        nb_tok_predicted = len(tokens_to_check)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens_to_check)
        self.complete_sequence(tokens_to_check)

        # Override from here
        tokens = cast(List[str], tokens_to_check.tokens)

        err_type = 0  # i.e. incompatible next type predicted
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_pitches = []

        # Init first note and current pitches if needed
        if previous_type == "Pitch":
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches.append(pitch_val)

        for i, token in enumerate(tokens[1:]):
            event_type, event_value = token.split("_")[0], token.split("_")[1]

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type in ["Bar", "TimeShift"]:  # reset
                    current_pitches = []
                elif event_type == "Pitch":
                    pitch_val = int(event_value)
                    if pitch_val in current_pitches:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches.append(pitch_val)
                elif event_type == "Program":  # reset
                    current_pitches = []
            # Bad token type
            else:
                err_type += 1
            previous_type = event_type

        return (err_type + err_note) / nb_tok_predicted
