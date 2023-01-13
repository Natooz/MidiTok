""" MIDI encoding method, similar to Compound Word
https://arxiv.org/abs/2101.02402

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event
from .utils import detect_chords
from .constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    CHORD_MAPS,
)


class CPWord(MIDITokenizer):
    r"""MIDI encoding method, similar to Compound Word
    https://arxiv.org/abs/2101.02402
    Each compound token will be a list of the form:
        (index. Token type)
        0. Family
        1. Bar/Position
        2. Pitch
        3. Velocity
        4. Duration
        (5. Program) optional, associated with notes (pitch/velocity/duration) or chords
        (6. Chord) optional, chords occurring with position tokens
        (7. Rest) optional, rest acting as a TimeShift token
        (8. Tempo) optional, occurring with position tokens
    This means a "compound token" can contain between 5 and 7 elements depending on
    your encoding parameters (additional tokens).
    (the choice of using indexes instead of dictionary with keys is to reduce the memory
    and storage usage for saved token files)

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...)
    :param pad: will include a PAD token, used when training a model with batch of sequences of
            unequal lengths, and usually at index 0 of the vocabulary. (default: True)
    :param sos_eos: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary.
            (default: False)
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary. (default: None)
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # Indexes of additional token types within a compound token
        additional_tokens["TimeSignature"] = False  # not compatible
        token_types = ["Family", "Position", "Pitch", "Velocity", "Duration"]
        add_tokens = ["Program", "Chord", "Rest", "Tempo"]
        for add_token in add_tokens:
            if additional_tokens[add_token]:
                token_types.append(add_token)
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation
        self.vocab_types_idx["Bar"] = 1  # same as position

        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            pad,
            sos_eos,
            mask,
            params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[List[int]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata["time_division"] / max(
            self.beat_res.values()
        )
        ticks_per_bar = self.current_midi_metadata["time_division"] * 4
        dur_bins = self.durations_ticks[self.current_midi_metadata["time_division"]]
        min_rest = (
            self.current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )
        tokens = []  # list of lists of tokens

        # Creates tokens
        previous_tick = -1
        previous_note_end = (
            track.notes[0].start + 1
        )  # so that no rest is created before the first note
        current_bar = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        for note in track.notes:
            # Bar / Position / (Tempo) / (Rest)
            if note.start != previous_tick:

                # (Rest)
                if (
                    self.additional_tokens["Rest"]
                    and note.start > previous_note_end
                    and note.start - previous_note_end >= min_rest
                ):
                    previous_tick = previous_note_end
                    rest_beat, rest_pos = divmod(
                        note.start - previous_tick,
                        self.current_midi_metadata["time_division"],
                    )
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        tokens.append(
                            self.__create_cp_token(
                                previous_note_end, rest=f"{rest_beat}.0", desc="Rest"
                            )
                        )
                        previous_tick += (
                            rest_beat * self.current_midi_metadata["time_division"]
                        )

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min(
                            [r[1] for r in self.rests], key=lambda x: abs(x - rest_pos)
                        )
                        tokens.append(
                            self.__create_cp_token(
                                previous_note_end,
                                rest=f"0.{rest_pos_temp}",
                                desc="Rest",
                            )
                        )
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    current_bar = previous_tick // ticks_per_bar

                # (Tempo)
                if self.additional_tokens["Tempo"]:
                    # If the current tempo is not the last one
                    if current_tempo_idx + 1 < len(
                        self.current_midi_metadata["tempo_changes"]
                    ):
                        # Will loop over incoming tempo changes
                        for tempo_change in self.current_midi_metadata["tempo_changes"][
                            current_tempo_idx + 1:
                        ]:
                            # If this tempo change happened before the current moment
                            if tempo_change.time <= note.start:
                                current_tempo = tempo_change.tempo
                                current_tempo_idx += (
                                    1  # update tempo value (might not change) and index
                                )
                            elif tempo_change.time > note.start:
                                break  # this tempo change is beyond the current time step, we break the loop

                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    tokens.append(
                        self.__create_cp_token(
                            (current_bar + i + 1) * ticks_per_bar, bar=True, desc="Bar"
                        )
                    )
                current_bar += nb_new_bars

                # Position
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                tokens.append(
                    self.__create_cp_token(
                        int(note.start),
                        pos=pos_index,
                        tempo=current_tempo
                        if self.additional_tokens["Tempo"]
                        else None,
                        desc="Position",
                    )
                )
                previous_tick = note.start

            # Note
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            dur_value = ".".join(map(str, self.durations[dur_index]))
            tokens.append(
                self.__create_cp_token(
                    int(note.start),
                    pitch=note.pitch,
                    vel=note.velocity,
                    dur=dur_value,
                    desc=f"{duration} ticks",
                )
            )
            previous_note_end = max(previous_note_end, note.end)

        tokens.sort(key=lambda x: x[0].time)

        # Adds chord tokens if specified
        if self.additional_tokens["Chord"] and not track.is_drum:
            chord_events = detect_chords(
                track.notes,
                self.current_midi_metadata["time_division"],
                self._first_beat_res,
            )
            count = 0
            for chord_event in chord_events:
                for e, cp_token in enumerate(tokens[count:]):
                    if (
                        cp_token[0].time == chord_event.time
                        and cp_token[0].desc == "Position"
                    ):
                        cp_token[self.vocab_types_idx["Chord"]] = self.vocab[
                            self.vocab_types_idx["Chord"]
                        ].event_to_token[f"Chord_{chord_event.value}"]
                        count = e
                        break

        # Convert the first element of each compound token from Event to int
        for cp_token in tokens:
            cp_token[0] = self.vocab[0].event_to_token[f"Family_{cp_token[0].value}"]

        return tokens

    def __create_cp_token(
        self,
        time: int,
        bar: bool = False,
        pos: int = None,
        pitch: int = None,
        vel: int = None,
        dur: str = None,
        chord: str = None,
        rest: str = None,
        tempo: int = None,
        program: int = None,
        desc: str = "",
    ) -> List[Union[Event, int]]:
        r"""Create a CP Word token, with the following structure:
            (index. Token type)
            0. Family
            1. Bar/Position
            2. Pitch
            3. Velocity
            4. Duration
            (5. Program) optional, associated with notes (pitch/velocity/duration) or chords
            (6. Chord) optional, chords occurring with position tokens
            (7. Rest) optional, rest acting as a TimeShift token
            (8. Tempo) optional, occurring with position tokens
        NOTE: the first Family token (first in list) will be given as an Event object to keep track
        of time easily so that other method can sort CP tokens afterwards.

        :param time: the current tick
        :param bar: True if this token represents a new bar occurring
        :param pos: the position index
        :param pitch: note pitch
        :param vel: note velocity
        :param dur: note duration
        :param chord: chord value
        :param rest: rest value
        :param tempo: tempo index
        :param program: a program number if you want to produce a Program CP token (read note above)
        :param desc: an optional argument for debug and used to spot position tokens in track_to_tokens
        :return: The compound token as a list of integers
        """
        cp_token_template = [
            Event(type_="Family", value="Metric", time=time, desc=desc),
            self.vocab[1].event_to_token["Ignore_None"],
            self.vocab[2].event_to_token["Ignore_None"],
            self.vocab[3].event_to_token["Ignore_None"],
            self.vocab[4].event_to_token["Ignore_None"],
        ]
        if self.additional_tokens["Program"]:
            cp_token_template.append(
                self.vocab[self.vocab_types_idx["Program"]].event_to_token[
                    "Ignore_None"
                ]
            )
        if self.additional_tokens["Chord"]:
            cp_token_template.append(
                self.vocab[self.vocab_types_idx["Chord"]].event_to_token["Ignore_None"]
            )
        if self.additional_tokens["Rest"]:
            cp_token_template.append(
                self.vocab[self.vocab_types_idx["Rest"]].event_to_token["Ignore_None"]
            )
        if self.additional_tokens["Tempo"]:
            cp_token_template.append(
                self.vocab[self.vocab_types_idx["Tempo"]].event_to_token["Ignore_None"]
            )

        if bar:
            cp_token_template[1] = self.vocab[1].event_to_token["Bar_None"]
        elif pos is not None:
            cp_token_template[1] = self.vocab[1].event_to_token[f"Position_{pos}"]
            if chord is not None:
                cp_token_template[self.vocab_types_idx["Chord"]] = self.vocab[
                    self.vocab_types_idx["Chord"]
                ].event_to_token[f"Chord_{chord}"]
            if tempo is not None:
                cp_token_template[self.vocab_types_idx["Tempo"]] = self.vocab[
                    self.vocab_types_idx["Tempo"]
                ].event_to_token[f"Tempo_{tempo}"]
        elif rest is not None:
            cp_token_template[self.vocab_types_idx["Rest"]] = self.vocab[
                self.vocab_types_idx["Rest"]
            ].event_to_token[f"Rest_{rest}"]
        elif pitch is not None:
            cp_token_template[0].value = "Note"
            cp_token_template[2] = self.vocab[2].event_to_token[f"Pitch_{pitch}"]
            cp_token_template[3] = self.vocab[3].event_to_token[f"Velocity_{vel}"]
            cp_token_template[4] = self.vocab[4].event_to_token[f"Duration_{dur}"]
            if program is not None:
                cp_token_template[self.vocab_types_idx["Program"]] = self.vocab[
                    self.vocab_types_idx["Program"]
                ].event_to_token[f"Program_{program}"]

        return cp_token_template

    def tokens_to_track(
        self,
        tokens: List[List[int]],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert (
            time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        events = self.tokens_to_events(tokens)

        ticks_per_sample = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below

        current_tick = 0
        current_bar = -1
        previous_note_end = 0
        for compound_token in events:
            token_family = compound_token[0].value
            if token_family == "Note":
                if any(tok.value == "None" for tok in compound_token[2:5]):
                    continue
                pitch = int(compound_token[2].value)
                vel = int(compound_token[3].value)
                duration = self._token_duration_to_ticks(
                    compound_token[4].value, time_division
                )
                instrument.notes.append(
                    Note(vel, pitch, current_tick, current_tick + duration)
                )
                previous_note_end = max(previous_note_end, current_tick + duration)
            elif token_family == "Metric":
                if compound_token[1].type == "Bar":
                    current_bar += 1
                    current_tick = current_bar * ticks_per_bar
                elif compound_token[1].type == "Position":  # i.e. its a position
                    if current_bar == -1:
                        current_bar = (
                            0  # as this Position token occurs before any Bar token
                        )
                    current_tick = (
                        current_bar * ticks_per_bar
                        + int(compound_token[1].value) * ticks_per_sample
                    )
                    if self.additional_tokens["Tempo"]:
                        tempo = int(compound_token[-1].value)
                        if tempo != tempo_changes[-1].tempo:
                            tempo_changes.append(TempoChange(tempo, current_tick))
                elif (
                    self.additional_tokens["Rest"]
                    and compound_token[self.vocab_types_idx["Rest"]].value != "None"
                ):
                    if (
                        current_tick < previous_note_end
                    ):  # if in case successive rest happen
                        current_tick = previous_note_end
                    beat, pos = map(
                        int,
                        compound_token[self.vocab_types_idx["Rest"]].value.split("."),
                    )
                    current_tick += beat * time_division + pos * ticks_per_sample
                    current_bar = current_tick // ticks_per_bar
        if len(tempo_changes) > 1:
            del tempo_changes[0]
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> List[Vocabulary]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is used as a padding index for training.

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print(
                "\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, "
                "_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m"
            )

        vocab = [
            Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            for _ in range(5)
        ]

        vocab[0].add_event("Family_Metric")
        vocab[0].add_event("Family_Note")

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/* time signature
        vocab[1].add_event("Ignore_None")
        vocab[1].add_event("Bar_None")
        vocab[1].add_event(f"Position_{i}" for i in range(nb_positions))

        # PITCH
        vocab[2].add_event("Ignore_None")
        vocab[2].add_event(f"Pitch_{i}" for i in self.pitch_range)

        # VELOCITY
        vocab[3].add_event("Ignore_None")
        vocab[3].add_event(f"Velocity_{i}" for i in self.velocities)

        # DURATION
        vocab[4].add_event("Ignore_None")
        vocab[4].add_event(
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        )

        # PROGRAM
        if self.additional_tokens["Program"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event("Ignore_None")
            vocab[-1].add_event(f"Program_{program}" for program in range(-1, 128))

        # CHORD
        if self.additional_tokens["Chord"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event("Ignore_None")
            vocab[-1].add_event(
                f"Chord_{i}" for i in range(3, 6)
            )  # non recognized chords (between 3 and 5 notes)
            vocab[-1].add_event(
                f"Chord_{chord_quality}" for chord_quality in CHORD_MAPS
            )

        # REST
        if self.additional_tokens["Rest"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event("Ignore_None")
            vocab[-1].add_event(
                f'Rest_{".".join(map(str, rest))}' for rest in self.rests
            )

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event("Ignore_None")
            vocab[-1].add_event(f"Tempo_{i}" for i in self.tempos)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        As with CP the tokens types are "merged", each state here corresponds to
        a "compound" token, which is characterized by the token types Program, Bar,
        Position/Chord/Tempo and Pitch/Velocity/Duration
        Here the combination of Pitch, Velocity and Duration tokens is represented by
        "Pitch" in the graph.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Bar"] = ["Position", "Bar"]
        dic["Position"] = ["Pitch"]
        dic["Pitch"] = ["Pitch", "Bar", "Position"]
        if self.additional_tokens["Chord"]:
            dic["Rest"] = ["Rest", "Position"]
            dic["Pitch"] += ["Rest"]
        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "Position", "Bar"]
            dic["Pitch"] += ["Rest"]

        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(
        self, tokens: List[List[int]], consider_pad: bool = False
    ) -> float:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a position token cannot have a value <= to the current position (it would go back in time)
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """

        def cp_token_type(tok: List[int]) -> Tuple[str, str]:
            family = self.vocab[0].token_to_event[tok[0]].split("_")[1]
            if family == "Note":
                return self.vocab[2].token_to_event[tok[2]].split("_")
            elif family == "Metric":
                bar_pos = self.vocab[1].token_to_event[tok[1]].split("_")
                if bar_pos[0] in ["Bar", "Position"]:
                    return bar_pos
                else:  # additional token
                    for i in range(1, 5):
                        decoded_token = (
                            self.vocab[-i].token_to_event[tok[-i]].split("_")
                        )
                        if decoded_token[0] != "Ignore":
                            return decoded_token
                raise RuntimeError("No token type found, unknown error")
            elif family == "None":
                return "PAD", "None"
            else:  # Program
                raise RuntimeError("No token type found, unknown error")

        err = 0
        previous_type = cp_token_type(tokens[0])[0]
        current_pos = -1
        current_pitches = []

        def check(tok: List[int]):
            nonlocal err, previous_type, current_pos, current_pitches
            token_type, token_value = cp_token_type(tok)
            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Bar":  # reset
                    current_pos = -1
                    current_pitches = []
                elif token_type == "Pitch":
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == "Position":
                    if int(token_value) <= current_pos and previous_type != "Rest":
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
                if previous_type == "PAD":
                    break
                check(token)
        return err / len(tokens)
