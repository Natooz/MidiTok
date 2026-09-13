"""BEAT (uniform temporal steps) tokenizer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import (
    DEFAULT_VELOCITY,
    MAX_STEP_EMBEDDING,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
)
from miditok.midi_tokenizer import MusicTokenizer

if TYPE_CHECKING:
    from pathlib import Path


class BEAT(MusicTokenizer):
    r"""
    BEAT (uniform temporal steps) tokenizer.

    BEAT is a flat tokenization representing time with a single *Step* token type on
    a uniform time grid, replacing the *Bar* and *Position* tokens of :ref:`REMI`.
    The index of a *Step* token is the event tick divided by the number of ticks per
    step, which is ``ticks_per_beat // config.max_num_pos_per_beat``. As BEAT does not
    support time signatures, this always amounts to one tick per step.
    Notes are represented as successions of *Pitch*, *Velocity* and *Duration* tokens.

    Introduced in `BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal
    Steps (Qian et al.) <https://arxiv.org/abs/2604.19532>`_.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object.
        BEAT accepts an additional param for tokenizer configuration,
        ``max_step_embedding``, which sets the maximum number of *Step* tokens
        (``Step_0`` to ``Step_{max_step_embedding - 1}``).
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful
        if the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer_config, params)

    def _tweak_config_before_creating_voc(self) -> None:
        if self.config.use_time_signatures:
            warnings.warn(
                "BEAT does not support time signatures, disabling them from the "
                "config.",
                stacklevel=2,
            )
            self.config.use_time_signatures = False
        if "max_step_embedding" not in self.config.additional_params:
            self.config.additional_params["max_step_embedding"] = MAX_STEP_EMBEDDING

    def _compute_ticks_per_pos(self, ticks_per_beat: int) -> int:
        return ticks_per_beat // self.config.max_num_pos_per_beat

    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        all_events = []
        previous_tick = -1
        previous_note_end = 0
        ticks_per_beat = time_division
        ticks_per_step = self._compute_ticks_per_pos(ticks_per_beat)

        for event in events:
            if event.type_.startswith("ACTrack"):
                all_events.append(event)
                continue
            if event.time != previous_tick:
                # (Rest)
                if (
                    self.config.use_rests
                    and event.time - previous_note_end >= self._min_rest(ticks_per_beat)
                ):
                    previous_tick = previous_note_end
                    rest_values = self._time_ticks_to_tokens(
                        event.time - previous_tick, ticks_per_beat, rest=True
                    )
                    # Add Rest events and increment previous_tick
                    for dur_value, dur_ticks in zip(*rest_values, strict=False):
                        all_events.append(
                            Event(
                                type_="Rest",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{event.time - previous_tick} ticks",
                            )
                        )
                        previous_tick += dur_ticks

                # Step
                all_events.append(
                    Event(
                        type_="Step",
                        value=event.time // ticks_per_step,
                        time=event.time,
                        desc=event.time,
                    )
                )
                previous_tick = event.time

            all_events.append(event)
            previous_note_end = self._previous_note_end_update(event, previous_note_end)

        return all_events

    @staticmethod
    def _previous_note_end_update(event: Event, previous_note_end: int) -> int:
        r"""
        Calculate max offset time of the notes encountered.

        uses Event field specified by event.type_ .
        """
        event_time = 0
        if event.type_ in {
            "Pitch",
            "PitchDrum",
            "PitchIntervalTime",
            "PitchIntervalChord",
        }:
            event_time = event.desc
        elif event.type_ in {
            "Program",
            "Tempo",
            "TimeSig",
            "Pedal",
            "PedalOff",
            "PitchBend",
            "Chord",
        }:
            event_time = event.time
        return max(previous_note_end, event_time)

    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i, tokens_i in enumerate(tokens):
            tokens[i] = tokens_i.tokens
        score = Score(self.time_division)
        dur_offset = 2 if self.config.use_velocities else 1

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes = []

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        current_track = None
        for si, seq in enumerate(tokens):
            ticks_per_beat = self._tpb_per_ts[TIME_SIGNATURE[1]]
            ticks_per_step = self._compute_ticks_per_pos(self.time_division)

            # Set tracking variables
            current_tick = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
            previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
            active_pedals = {}

            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type.startswith("Program"):
                            current_program = int(tok_val)
                            if current_program == -1:
                                is_drum, current_program = True, 0
                            break
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )
            current_track_use_duration = (
                current_program in self.config.use_note_duration_programs
            )

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")
                if tok_type == "Step":
                    current_tick = int(tok_val) * ticks_per_step
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                elif tok_type in {
                    "Pitch",
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                }:
                    if tok_type in {"Pitch", "PitchDrum"}:
                        pitch = int(tok_val)
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                    else:  # PitchIntervalChord
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                    if (
                        not self.config.pitch_range[0]
                        <= pitch
                        <= self.config.pitch_range[1]
                    ):
                        continue

                    # We update previous_pitch_onset and previous_pitch_chord even if
                    # the try fails.
                    if tok_type != "PitchIntervalChord":
                        previous_pitch_onset[current_program] = pitch
                    previous_pitch_chord[current_program] = pitch

                    try:
                        if self.config.use_velocities:
                            vel_type, vel = seq[ti + 1].split("_")
                        else:
                            vel_type, vel = "Velocity", DEFAULT_VELOCITY
                        if current_track_use_duration:
                            dur_type, dur = seq[ti + dur_offset].split("_")
                        else:
                            dur_type = "Duration"
                            dur = int(
                                self.config.default_note_duration * ticks_per_beat
                            )
                        if vel_type == "Velocity" and dur_type == "Duration":
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            new_note = Note(
                                current_tick,
                                dur,
                                pitch,
                                int(vel),
                            )
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                            previous_note_end = max(
                                previous_note_end, current_tick + dur
                            )
                    except IndexError:
                        # A well constituted sequence should not raise an exception
                        # However with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass
                elif tok_type == "Program":
                    current_program = int(tok_val)
                    current_track_use_duration = (
                        current_program in self.config.use_note_duration_programs
                    )
                    if (
                        not self.config.one_token_stream_for_programs
                        and self.config.program_changes
                    ):
                        if current_program != -1:
                            current_track.program = current_program
                        else:
                            current_track.program = 0
                            current_track.is_drum = True
                elif tok_type == "Tempo":
                    if si == 0:
                        tempo_changes.append(Tempo(current_tick, float(tok_val)))
                    previous_note_end = max(previous_note_end, current_tick)
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                                seq[ti + 1].split("_")[1]
                            ]
                            # Add instrument if it doesn't exist, can happen for the
                            # first tokens
                            new_pedal = Pedal(current_tick, duration)
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_prog not in active_pedals:
                        active_pedals[pedal_prog] = current_tick
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        new_pedal = Pedal(
                            active_pedals[pedal_prog],
                            current_tick - active_pedals[pedal_prog],
                        )
                        if self.config.one_token_stream_for_programs:
                            check_inst(pedal_prog)
                            tracks[pedal_prog].pedals.append(new_pedal)
                        else:
                            current_track.pedals.append(new_pedal)
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.config.one_token_stream_for_programs:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_track.pitch_bends.append(new_pitch_bend)

            # Add current_inst to score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        # Add global events to the score
        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = [TimeSignature(0, *TIME_SIGNATURE)]

        return score

    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        :return: the vocabulary as a list of string.
        """
        vocab = [
            f"Step_{i}"
            for i in range(self.config.additional_params["max_step_embedding"])
        ]

        # NoteOn/NoteOff/Velocity
        self._add_note_tokens_to_vocab_list(vocab)

        # Add additional tokens
        self._add_additional_tokens_to_vocab_list(vocab)

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic: dict[str, set[str]] = {}

        if self.config.use_programs:
            first_note_token_type = (
                "Pitch" if self.config.program_changes else "Program"
            )
            dic["Program"] = {"Pitch"}
        else:
            first_note_token_type = "Pitch"
        if self.config.use_velocities:
            dic["Pitch"] = {"Velocity"}
            dic["Velocity"] = (
                {"Duration"}
                if self.config.using_note_duration_tokens
                else {first_note_token_type, "Step"}
            )
        elif self.config.using_note_duration_tokens:
            dic["Pitch"] = {"Duration"}
        else:
            dic["Pitch"] = {first_note_token_type, "Step"}
        if self.config.using_note_duration_tokens:
            dic["Duration"] = {first_note_token_type, "Step"}
        dic["Step"] = {"Step", first_note_token_type}
        if self.config.use_pitch_intervals:
            for token_type in ("PitchIntervalTime", "PitchIntervalChord"):
                dic[token_type] = (
                    {"Velocity"}
                    if self.config.use_velocities
                    else {"Duration"}
                    if self.config.using_note_duration_tokens
                    else {
                        first_note_token_type,
                        "PitchIntervalTime",
                        "PitchIntervalChord",
                        "Step",
                    }
                )
                if (
                    self.config.use_programs
                    and self.config.one_token_stream_for_programs
                ):
                    dic["Program"].add(token_type)
                else:
                    if self.config.using_note_duration_tokens:
                        dic["Duration"].add(token_type)
                    elif self.config.use_velocities:
                        dic["Velocity"].add(token_type)
                    else:
                        dic["Pitch"].add(token_type)
                    dic["Step"].add(token_type)
        if self.config.program_changes:
            dic[
                "Duration"
                if self.config.using_note_duration_tokens
                else "Velocity"
                if self.config.use_velocities
                else first_note_token_type
            ].add("Program")

        if self.config.use_chords:
            dic["Chord"] = {first_note_token_type}
            dic["Step"] |= {"Chord"}
            if self.config.use_programs:
                dic["Program"].add("Chord")
            if self.config.use_pitch_intervals:
                dic["Chord"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_tempos:
            dic["Step"] |= {"Tempo"}
            dic["Tempo"] = {first_note_token_type, "Step"}
            if self.config.use_chords:
                dic["Tempo"] |= {"Chord"}
            if self.config.use_rests:
                dic["Tempo"].add("Rest")  # only for first token
            if self.config.use_pitch_intervals:
                dic["Tempo"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_sustain_pedals:
            dic["Step"].add("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = {"Duration"}
                if self.config.using_note_duration_tokens:
                    dic["Duration"].add("Pedal")
                elif self.config.use_velocities:
                    dic["Duration"] = {first_note_token_type, "Step"}
                    dic["Velocity"].add("Pedal")
                else:
                    dic["Duration"] = {first_note_token_type, "Step"}
                    dic["Pitch"].add("Pedal")
            else:
                dic["PedalOff"] = {
                    "Pedal",
                    "PedalOff",
                    first_note_token_type,
                    "Step",
                }
                dic["Pedal"] = {"Pedal", first_note_token_type, "Step"}
                dic["Step"].add("PedalOff")
            if self.config.use_chords:
                dic["Pedal"].add("Chord")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].add("Chord")
                    dic["Chord"].add("PedalOff")
            if self.config.use_rests:
                dic["Pedal"].add("Rest")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].add("Rest")
            if self.config.use_tempos:
                dic["Tempo"].add("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["Tempo"].add("PedalOff")
            if self.config.use_pitch_intervals:
                if self.config.sustain_pedal_duration:
                    dic["Duration"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                else:
                    dic["Pedal"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                    dic["PedalOff"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_pitch_bends:
            # As a Program token will precede PitchBend otherwise
            # Else no need to add Program as its already in
            dic["PitchBend"] = {first_note_token_type, "Step"}
            if self.config.use_programs and not self.config.program_changes:
                dic["Program"].add("PitchBend")
            else:
                dic["Step"].add("PitchBend")
                if self.config.use_tempos:
                    dic["Tempo"].add("PitchBend")
                if self.config.use_sustain_pedals:
                    dic["Pedal"].add("PitchBend")
                    if self.config.sustain_pedal_duration:
                        dic["Duration"].add("PitchBend")
                    else:
                        dic["PedalOff"].add("PitchBend")
            if self.config.use_chords:
                dic["PitchBend"].add("Chord")
            if self.config.use_rests:
                dic["PitchBend"].add("Rest")

        if self.config.use_rests:
            dic["Rest"] = {"Rest", first_note_token_type, "Step"}
            dic["Step"].add("Rest")
            dic[
                "Duration"
                if self.config.using_note_duration_tokens
                else "Velocity"
                if self.config.use_velocities
                else "Pitch"
            ].add("Rest")
            if (
                not self.config.using_note_duration_tokens
                and not self.config.use_velocities
            ):
                for token_type in (
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                ):
                    if token_type in dic:
                        dic[token_type].add("Rest")
            if self.config.use_chords:
                dic["Rest"] |= {"Chord"}
            if self.config.use_tempos:
                dic["Rest"].add("Tempo")
            if self.config.use_sustain_pedals:
                dic["Rest"].add("Pedal")
                if self.config.sustain_pedal_duration:
                    dic["Duration"].add("Rest")
                else:
                    dic["Rest"].add("PedalOff")
                    dic["PedalOff"].add("Rest")
            if self.config.use_pitch_bends:
                dic["Rest"].add("PitchBend")
            if self.config.use_pitch_intervals:
                dic["Rest"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.program_changes:
            for token_type in (
                "Step",
                "Rest",
                "PitchBend",
                "Pedal",
                "PedalOff",
                "Tempo",
                "Chord",
            ):
                if token_type in dic:
                    dic["Program"].add(token_type)
                    dic[token_type].add("Program")

        if self.config.use_pitchdrum_tokens:
            dic["PitchDrum"] = dic["Pitch"]
            for values in dic.values():
                if "Pitch" in values:
                    values.add("PitchDrum")

        return dic

    def _tokens_errors(self, tokens: list[str]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        This method checks token types successions and duplicated notes. A *Step*
        token resets the pitches played at the current position, as each step is a new
        temporal position.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err_type = 0  # i.e. incompatible next type predicted
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_program = 0
        current_pitches = {p: [] for p in self.config.programs}
        previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
        previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
        note_tokens_types = ["Pitch", "NoteOn", "PitchDrum"]
        if self.config.use_pitch_intervals:
            note_tokens_types += ["PitchIntervalTime", "PitchIntervalChord"]

        # Init first note and current pitches if needed
        if previous_type in note_tokens_types:
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches[current_program].append(pitch_val)

        for token in tokens[1:]:
            event_type, event_value = token.split("_")

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type in ["Step", "Rest"]:
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in note_tokens_types:
                    if event_type in {"Pitch", "NoteOn", "PitchDrum"}:
                        pitch_val = int(event_value)
                        previous_pitch_onset[current_program] = pitch_val
                        previous_pitch_chord[current_program] = pitch_val
                    elif event_type == "PitchIntervalTime":
                        pitch_val = previous_pitch_onset[current_program] + int(
                            event_value
                        )
                        previous_pitch_onset[current_program] = pitch_val
                        previous_pitch_chord[current_program] = pitch_val
                    else:  # PitchIntervalChord
                        pitch_val = previous_pitch_chord[current_program] + int(
                            event_value
                        )
                        previous_pitch_chord[current_program] = pitch_val
                    if (
                        self.config.remove_duplicated_notes
                        and pitch_val in current_pitches[current_program]
                    ):
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches[current_program].append(pitch_val)
                elif event_type == "Program":  # reset
                    current_program = int(event_value)
            # Bad token type
            else:
                err_type += 1
            previous_type = event_type

        return err_type + err_note
