from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from ..midi_tokenizer import MIDITokenizer, _in_as_seq
from ..classes import TokSequence, Event
from ..constants import (
    TIME_DIVISION,
    TIME_SIGNATURE,
    TEMPO,
    MIDI_INSTRUMENTS,
)


class Octuple(MIDITokenizer):
    r"""Introduced with `MusicBert (Zeng et al.) <https://arxiv.org/abs/2106.05630>`_,
    the idea of Octuple is to use embedding pooling so that each pooled embedding
    represents a single note. Tokens (*Pitch*, *Velocity*...) are first independently
    converted to embeddings which are then merged (pooled) into a single one.
    Each pooled token will be a list of the form (index: Token type):
    * 0: Pitch
    * 1: Velocity
    * 2: Duration
    * 3: Position
    * 4: Bar
    * (+ Optional) Program
    * (+ Optional) Tempo
    * (+ Optional) TimeSignature

    Its considerably reduces the sequence lengths, while handling multitrack.
    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can be
    very delicate. Hence, we do not recommend this tokenization for generation with small models.

    **Notes:**
    * Tokens are first sorted by time, then track, then pitch values.
    * Tracks with the same *Program* will be merged.
    * When decoding multiple token sequences (of multiple tracks), i.e. when `config.use_programs` is False,
    only the tempos and time signatures of the first sequence will be decoded for the whole MIDI.
    """

    def _tweak_config_before_creating_voc(self):
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.delete_equal_successive_tempo_changes = True
        self.config.program_changes = False

        # used in place of positional encoding
        # This attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = 60

        token_types = ["Pitch", "Velocity", "Duration", "Position", "Bar"]
        if self.config.use_programs:
            token_types.append("Program")
        if self.config.use_tempos:
            token_types.append("Tempo")
        if self.config.use_time_signatures:
            token_types.append("TimeSig")
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation

    def _add_time_events(self, events: List[Event]) -> List[List[Event]]:
        r"""
        Takes a sequence of note events (containing optionally Chord, Tempo and TimeSignature tokens),
        and insert (not inplace) time tokens (TimeShift, Rest) to complete the sequence.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Position
            4: Bar
            (5: Program)
            (6: Tempo)
            (7: TimeSignature)

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.config.beat_res.values())

        # Add time events
        all_events = []
        current_bar = 0
        current_bar_from_ts_time = 0
        current_tick_from_ts_time = 0
        current_pos = 0
        previous_tick = 0
        current_time_sig = TIME_SIGNATURE
        current_tempo = TEMPO
        current_program = None
        ticks_per_bar = self._compute_ticks_per_bar(
            TimeSignature(*current_time_sig, 0), time_division
        )
        for e, event in enumerate(events):
            # Set current bar and position
            # This is done first, as we need to compute these values with the current ticks_per_bar,
            # which might change if the current event is a TimeSig
            if event.time != previous_tick:
                elapsed_tick = event.time - current_tick_from_ts_time
                current_bar = current_bar_from_ts_time + elapsed_tick // ticks_per_bar
                current_pos = int((elapsed_tick % ticks_per_bar) / ticks_per_sample)
                previous_tick = event.time

            if event.type == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                current_bar_from_ts_time = current_bar
                current_tick_from_ts_time = previous_tick
                ticks_per_bar = self._compute_ticks_per_bar(
                    TimeSignature(*current_time_sig, event.time), time_division
                )
            elif event.type == "Tempo":
                current_tempo = event.value
            elif event.type == "Program":
                current_program = event.value
            elif event.type == "Pitch" and e + 2 < len(events):
                new_event = [
                    Event(type="Pitch", value=event.value, time=event.time),
                    Event(type="Velocity", value=events[e + 1].value, time=event.time),
                    Event(type="Duration", value=events[e + 2].value, time=event.time),
                    Event(type="Position", value=current_pos, time=event.time),
                    Event(type="Bar", value=current_bar, time=event.time),
                ]
                if self.config.use_programs:
                    new_event.append(Event("Program", current_program))
                if self.config.use_tempos:
                    new_event.append(Event(type="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
                all_events.append(new_event)

        return all_events

    def _midi_to_tokens(
        self, midi: MidiFile, *args, **kwargs
    ) -> Union[TokSequence, List[TokSequence]]:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        The workflow of this method is as follows: the events (Pitch, Velocity, Tempo, TimeSignature...) are
        gathered into a list, then the time events are added. If `one_token_stream` is true, all events of all tracks
        are treated all at once, otherwise the events of each track are treated independently.

        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Position
            4: Bar
            (5: Program)
            (6: Tempo)
            (7: TimeSignature)

        :param midi: the MIDI object to convert
        :return: sequences of tokens
        """
        # Check bar embedding limit, update if needed
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.config.additional_params["max_bar_embedding"] < nb_bars:
            for i in range(self.config.additional_params["max_bar_embedding"], nb_bars):
                self.add_to_vocab(f"Bar_{i}", 4)
            self.config.additional_params["max_bar_embedding"] = nb_bars

        return super()._midi_to_tokens(midi, *args, **kwargs)

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[
            Union[TokSequence, List, np.ndarray, Any],
            List[Union[TokSequence, List, np.ndarray, Any]],
        ],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param programs: programs of the tracks. If none is given, will default to piano, program 0. (default: None)
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.one_token_stream:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
            time_division % max(self.config.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.config.beat_res.values())}"
        ticks_per_sample = time_division // max(self.config.beat_res.values())

        # RESULTS
        instruments: Dict[int, Instrument] = {}
        tempo_changes = [TempoChange(TEMPO, -1)]
        time_signature_changes = [TimeSignature(*TIME_SIGNATURE, 0)]
        ticks_per_bar = self._compute_ticks_per_bar(
            time_signature_changes[0], time_division
        )  # init

        current_bar_from_ts_time = 0
        current_tick_from_ts_time = 0
        current_program = 0
        for si, seq in enumerate(tokens):
            # Set track / sequence program if needed
            if not self.one_token_stream:
                ticks_per_bar = self._compute_ticks_per_bar(
                    time_signature_changes[0], time_division
                )
                if programs is not None:
                    current_program = -1 if programs[si][1] else programs[si][0]

            # Decode tokens
            for time_step in seq:
                nb_tok_to_check = 6 if self.config.use_programs else 5
                if any(
                    tok.split("_")[1] == "None" for tok in time_step[:nb_tok_to_check]
                ):
                    continue  # Either padding, mask: error of prediction or end of sequence anyway

                # Note attributes
                pitch = int(time_step[0].split("_")[1])
                vel = int(time_step[1].split("_")[1])
                duration = self._token_duration_to_ticks(
                    time_step[2].split("_")[1], time_division
                )
                if self.config.use_programs:
                    current_program = int(time_step[5].split("_")[1])

                # Time values
                event_pos = int(time_step[3].split("_")[1])
                event_bar = int(time_step[4].split("_")[1])
                current_tick = (
                    current_tick_from_ts_time
                    + (event_bar - current_bar_from_ts_time) * ticks_per_bar
                    + event_pos * ticks_per_sample
                )

                # Append the created note
                if current_program not in instruments.keys():
                    instruments[current_program] = Instrument(
                        program=0 if current_program == -1 else current_program,
                        is_drum=current_program == -1,
                        name="Drums"
                        if current_program == -1
                        else MIDI_INSTRUMENTS[current_program]["name"],
                    )
                instruments[current_program].notes.append(
                    Note(vel, pitch, current_tick, current_tick + duration)
                )

                # Tempo, adds a TempoChange if necessary
                if (
                    si == 0
                    and self.config.use_tempos
                    and time_step[self.vocab_types_idx["Tempo"]].split("_")[1] != "None"
                ):
                    tempo = float(
                        time_step[self.vocab_types_idx["Tempo"]].split("_")[1]
                    )
                    if tempo != tempo_changes[-1].tempo:
                        tempo_changes.append(TempoChange(tempo, current_tick))

                # Time Signature, adds a TimeSignatureChange if necessary
                if (
                    self.config.use_time_signatures
                    and time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]
                    != "None"
                ):
                    num, den = self._parse_token_time_signature(
                        time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]
                    )
                    if (
                        num != time_signature_changes[-1].numerator
                        and den != time_signature_changes[-1].denominator
                    ):
                        time_sig = TimeSignature(num, den, current_tick)
                        if si == 0:
                            time_signature_changes.append(time_sig)
                        current_bar_from_ts_time = event_bar
                        current_tick_from_ts_time = current_tick
                        ticks_per_bar = self._compute_ticks_per_bar(
                            time_sig, time_division
                        )

        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        if len(time_signature_changes) > 1:
            del time_signature_changes[0]  # delete mocked time signature change
        time_signature_changes[0].time = 0

        # create MidiFile
        midi.instruments = list(instruments.values())
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

    def _create_base_vocabulary(self) -> List[List[str]]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = [[] for _ in range(5)]

        # PITCH
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # VELOCITY
        vocab[1] += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab[2] += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # POSITION
        max_nb_beats = max(
            map(lambda ts: ceil(4 * ts[0] / ts[1]), self.time_signatures)
        )
        nb_positions = max(self.config.beat_res.values()) * max_nb_beats
        vocab[3] += [f"Position_{i}" for i in range(nb_positions)]

        # BAR (positional encoding)
        vocab[4] += [
            f"Bar_{i}"
            for i in range(self.config.additional_params["max_bar_embedding"])
        ]

        # PROGRAM
        if self.config.use_programs:
            vocab.append([f"Program_{i}" for i in self.config.programs])

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            vocab.append([f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures])

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Not relevant for Octuple.

        :return: the token types transitions dictionary
        """
        return {}  # not relevant for Octuple

    @_in_as_seq()
    def tokens_errors(
        self, tokens: Union[TokSequence, List, np.ndarray, Any]
    ) -> Union[float, List[float]]:
        r"""Checks if a sequence of tokens is made of good token values and
        returns the error ratio (lower is better).
        The token types are always the same in Octuple so this methods only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :return: the error ratio (lower is better)
        """
        # If list of TokSequence -> recursive
        if isinstance(tokens, list):
            return [self.tokens_errors(tok_seq) for tok_seq in tokens]

        err = 0
        current_bar = current_pos = -1
        current_pitches = {p: [] for p in self.config.programs}
        current_program = 0

        for token in tokens.tokens:
            if any(tok.split("_")[1] == "None" for tok in token):
                err += 1
                continue
            has_error = False
            bar_value = int(token[4].split("_")[1])
            pos_value = int(token[3].split("_")[1])
            pitch_value = int(token[0].split("_")[1])
            if self.config.use_programs:
                current_program = int(token[5].split("_")[1])

            # Bar
            if bar_value < current_bar:
                has_error = True
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = -1
                current_pitches = {p: [] for p in self.config.programs}

            # Position
            if pos_value < current_pos:
                has_error = True
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = {p: [] for p in self.config.programs}

            # Pitch
            if pitch_value in current_pitches[current_program]:
                has_error = True
            else:
                current_pitches[current_program].append(pitch_value)

            if has_error:
                err += 1

        return err / len(tokens)
