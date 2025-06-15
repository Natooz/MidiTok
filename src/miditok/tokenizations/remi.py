"""REMI (Revamped MIDI) tokenizer."""

from __future__ import annotations

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
    ADD_TRAILING_BARS,
    DEFAULT_VELOCITY,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
    USE_BAR_END_TOKENS,
)
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class REMI(MusicTokenizer):
    r"""
    REMI (Revamped MIDI) tokenizer.

    Introduced with the `Pop Music Transformer (Huang and Yang) <https://dl.acm.org/doi/10.1145/3394171.3413671>`_,
    REMI represents notes as successions of *Pitch*, *Velocity* and
    *Duration* tokens, and time with *Bar* and *Position* tokens. A *Bar* token
    indicate that a new bar is beginning, and *Position* the current position within
    the current bar. The number of positions is determined by the ``beat_res``
    argument, the maximum value will be used as resolution.
    With the *Program* and *TimeSignature* additional tokens enables, this class is
    equivalent to REMI+. REMI+ is an extended version of :ref:`REMI` (Huang and Yang)
    for general multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`_, which handle multiple
    instruments by adding *Program* tokens before the *Pitch* ones.

    **Note:** in the original paper, the tempo information is represented as the
    succession of two token types: a *TempoClass* indicating if the tempo is fast or
    slow, and a *TempoValue* indicating its value. MidiTok only uses one *Tempo* token
    for its value (see :ref:`Additional tokens`).
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when
    `config.use_programs` is False, only the tempos and time signatures of the first
    sequence will be decoded for the whole music.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object.
        REMI accepts additional_params for tokenizer configuration:
        | - max_bar_embedding (desciption below);
        | - use_bar_end_tokens -- if set will add Bar_End tokens at the end of each Bar;
        | - add_trailing_bars -- will add tokens for trailing empty Bars, if they are
        present in source symbolic music data. Applicable to :ref:`REMI`. This flag is
        very useful in applications where we need bijection between Bars is source and
        tokenized representations, same lengths, anacrusis detection etc.
        False by default, thus trailing bars are omitted.
    :param max_bar_embedding: Maximum number of bars ("Bar_0", "Bar_1",...,
        "Bar_{num_bars-1}"). If None passed, creates "Bar_None" token only in
        vocabulary for Bar token. Has less priority than setting this parameter via
        TokenizerConfig additional_params
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful
        if the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        max_bar_embedding: int | None = None,
        params: str | Path | None = None,
    ) -> None:
        if (
            max_bar_embedding is not None
            and tokenizer_config is not None
            and "max_bar_embedding" not in tokenizer_config.additional_params
        ):
            tokenizer_config.additional_params["max_bar_embedding"] = max_bar_embedding
        super().__init__(tokenizer_config, params)

    def _tweak_config_before_creating_voc(self) -> None:
        # In case the tokenizer has been created without specifying any config or
        # params file path
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = None
        if "use_bar_end_tokens" not in self.config.additional_params:
            self.config.additional_params["use_bar_end_tokens"] = USE_BAR_END_TOKENS
        if "add_trailing_bars" not in self.config.additional_params:
            self.config.additional_params["add_trailing_bars"] = ADD_TRAILING_BARS

    def _compute_ticks_per_pos(self, ticks_per_beat: int) -> int:
        return ticks_per_beat // self.config.max_num_pos_per_beat

    def _compute_ticks_per_units(
        self, time: int, current_time_sig: Sequence[int], time_division: int
    ) -> tuple[int, int, int]:
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(time, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
        return ticks_per_bar, ticks_per_beat, ticks_per_pos

    def _add_new_bars(
        self,
        until_time: int,
        event_type: str,
        all_events: Sequence[Event],
        current_bar: int,
        bar_at_last_ts_change: int,
        tick_at_last_ts_change: int,
        tick_at_current_bar: int,
        current_time_sig: tuple[int, int],
        ticks_per_bar: int,
    ) -> tuple[int, int]:
        num_new_bars = (
            bar_at_last_ts_change
            + self._units_between(tick_at_last_ts_change, until_time, ticks_per_bar)
            - current_bar
        )
        for i in range(num_new_bars):
            current_bar += 1
            tick_at_current_bar = (
                tick_at_last_ts_change
                + (current_bar - bar_at_last_ts_change) * ticks_per_bar
            )
            if self.config.additional_params["use_bar_end_tokens"] and current_bar > 0:
                all_events.append(
                    Event(
                        type_="Bar",
                        value="End",
                        time=tick_at_current_bar - 1,
                        desc=0,
                    )
                )
            all_events.append(
                Event(
                    type_="Bar",
                    value=str(current_bar + i)
                    if self.config.additional_params["max_bar_embedding"] is not None
                    else "None",
                    time=tick_at_current_bar,
                    desc=0,
                )
            )
            # Add a TimeSignature token, except for the last new Bar token
            # if the current event is a TS
            if self.config.use_time_signatures and not (
                event_type == "TimeSig" and i + 1 == num_new_bars
            ):
                all_events.append(
                    Event(
                        type_="TimeSig",
                        value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        time=tick_at_current_bar,
                        desc=0,
                    )
                )
        return current_bar, tick_at_current_bar

    def _add_position_event(
        self,
        event: Event,
        all_events: list[Event],
        tick_at_current_bar: int,
        ticks_per_pos: int,
    ) -> None:
        pos_index = self._units_between(tick_at_current_bar, event.time, ticks_per_pos)
        # if event in TOKEN_TYPES_AC
        all_events.append(
            Event(
                type_="Position",
                value=pos_index,
                time=event.time,
                desc=event.time,
            )
        )

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
        # Add time events
        all_events = []
        current_bar = -1
        bar_at_last_ts_change = 0
        previous_tick = -1
        previous_note_end = 0
        tick_at_last_ts_change = tick_at_current_bar = 0

        # Determine time signature and compute ticks per entites
        current_time_sig, time_sig_time = TIME_SIGNATURE, 0

        # First look for a TimeSig token, if any is given at tick 0, to update
        # current_time_sig
        if self.config.use_time_signatures:
            for event in events:
                # There should be a TimeSig token at tick 0
                if event.type_ == "TimeSig":
                    current_time_sig = self._parse_token_time_signature(event.value)
                    time_sig_time = event.time
                    break
        ticks_per_bar, ticks_per_beat, ticks_per_pos = self._compute_ticks_per_units(
            time_sig_time, current_time_sig, time_division
        )

        # Add the time events
        for ei, event in enumerate(events):
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
                    for dur_value, dur_ticks in zip(*rest_values):
                        all_events.append(
                            Event(
                                type_="Rest",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{event.time - previous_tick} ticks",
                            )
                        )
                        previous_tick += dur_ticks
                    # We update current_bar and tick_at_current_bar here without
                    # creating Bar tokens
                    real_current_bar = bar_at_last_ts_change + self._units_between(
                        tick_at_last_ts_change, previous_tick, ticks_per_bar
                    )
                    if real_current_bar > current_bar:
                        # In case we instantly begin with a Rest,
                        # we need to update current_bar
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (
                            real_current_bar - current_bar
                        ) * ticks_per_bar
                        current_bar = real_current_bar

                # Bar
                current_bar, tick_at_current_bar = self._add_new_bars(
                    event.time,
                    event.type_,
                    all_events,
                    current_bar,
                    bar_at_last_ts_change,
                    tick_at_last_ts_change,
                    tick_at_current_bar,
                    current_time_sig,
                    ticks_per_bar,
                )

                # Position
                if event.type_ != "TimeSig" and not event.type_.startswith("ACBar"):
                    self._add_position_event(
                        event, all_events, tick_at_current_bar, ticks_per_pos
                    )

                previous_tick = event.time

            # Update time signature time variables, after adjusting the time (above)
            if event.type_ == "TimeSig":
                bar_at_last_ts_change += self._units_between(
                    tick_at_last_ts_change, event.time, ticks_per_bar
                )
                tick_at_last_ts_change = event.time
                current_time_sig = self._parse_token_time_signature(event.value)
                ticks_per_bar, ticks_per_beat, ticks_per_pos = (
                    self._compute_ticks_per_units(
                        event.time, current_time_sig, time_division
                    )
                )
                # We decrease the previous tick so that a Position token is enforced
                # for the next event
                previous_tick -= 1

            all_events.append(event)
            # Adds a Position token if the current event is a bar-level attribute
            # control and the next one is at the same position, as the position token
            # wasn't added previously.
            if (
                event.type_.startswith("ACBar")
                and not events[ei + 1].type_.startswith("ACBar")
                and event.time == events[ei + 1].time
            ):
                self._add_position_event(
                    event, all_events, tick_at_current_bar, ticks_per_pos
                )

            # Update max offset time of the notes encountered
            previous_note_end = self._previous_note_end_update(event, previous_note_end)

        if (
            previous_note_end > previous_tick
            and self.config.additional_params["add_trailing_bars"]
        ):
            # there are some trailing bars
            _ = self._add_new_bars(
                previous_note_end,
                event.type_,
                all_events,
                current_bar,
                bar_at_last_ts_change,
                tick_at_last_ts_change,
                tick_at_current_bar,
                current_time_sig,
                ticks_per_bar,
            )
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

    @staticmethod
    def _units_between(start_tick: int, end_tick: int, ticks_per_unit: int) -> int:
        return (end_tick - start_tick) // ticks_per_unit

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
        tempo_changes, time_signature_changes = [], []

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
            # First look for the first time signature if needed
            if si == 0:
                if self.config.use_time_signatures:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type == "TimeSig":
                            time_signature_changes.append(
                                TimeSignature(
                                    0, *self._parse_token_time_signature(tok_val)
                                )
                            )
                            break
                        if tok_type in [
                            "Pitch",
                            "PitchDrum",
                            "Velocity",
                            "Duration",
                            "PitchBend",
                            "Pedal",
                        ]:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[-1]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, score.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)

            # Set tracking variables
            current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
            current_bar = -1
            bar_at_last_ts_change = 0
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
                if token == "Bar_None":
                    current_bar += 1
                    if current_bar > 0:
                        current_tick = tick_at_current_bar + ticks_per_bar
                    tick_at_current_bar = current_tick
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                    real_current_bar = bar_at_last_ts_change + self._units_between(
                        tick_at_last_ts_change, current_tick, ticks_per_bar
                    )
                    if real_current_bar > current_bar:
                        # In case we instantly begin with a Rest,
                        # we need to update current_bar
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (
                            real_current_bar - current_bar
                        ) * ticks_per_bar
                        current_bar = real_current_bar
                elif tok_type == "Position":
                    if current_bar == -1:
                        # as this Position token occurs before any Bar token
                        current_bar = 0
                    current_tick = tick_at_current_bar + int(tok_val) * ticks_per_pos
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
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if (
                        num != current_time_sig.numerator
                        or den != current_time_sig.denominator
                    ):
                        current_time_sig = TimeSignature(current_tick, num, den)
                        if si == 0:
                            time_signature_changes.append(current_time_sig)
                        tick_at_last_ts_change = tick_at_current_bar  # == current_tick
                        bar_at_last_ts_change = current_bar
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, score.ticks_per_quarter
                        )
                        ticks_per_beat = self._tpb_per_ts[den]
                        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
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
        score.time_signatures = time_signature_changes

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
        vocab = []

        # Bar
        if self.config.additional_params["max_bar_embedding"] is not None:
            vocab += [
                f"Bar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
        else:
            vocab += ["Bar_None"]
        if self.config.additional_params["use_bar_end_tokens"]:
            vocab.append("Bar_End")

        # NoteOn/NoteOff/Velocity
        self._add_note_tokens_to_vocab_list(vocab)

        # Position
        # self.time_division is equal to the maximum possible ticks/beat value.
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab += [f"Position_{i}" for i in range(num_positions)]

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
                else {first_note_token_type, "Position", "Bar"}
            )
        elif self.config.using_note_duration_tokens:
            dic["Pitch"] = {"Duration"}
        else:
            dic["Pitch"] = {first_note_token_type, "Bar", "Position"}
        if self.config.using_note_duration_tokens:
            dic["Duration"] = {first_note_token_type, "Position", "Bar"}
        dic["Bar"] = {"Position", "Bar"}
        dic["Position"] = {first_note_token_type}
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
                        "Bar",
                        "Position",
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
                    dic["Position"].add(token_type)
        if self.config.program_changes:
            dic[
                "Duration"
                if self.config.using_note_duration_tokens
                else "Velocity"
                if self.config.use_velocities
                else first_note_token_type
            ].add("Program")
            # The first bar may be empty but the Program token will still be present
            if self.config.additional_params["use_bar_end_tokens"]:
                dic["Program"].add("Bar")

        if self.config.use_chords:
            dic["Chord"] = {first_note_token_type}
            dic["Position"] |= {"Chord"}
            if self.config.use_programs:
                dic["Program"].add("Chord")
            if self.config.use_pitch_intervals:
                dic["Chord"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_tempos:
            dic["Position"] |= {"Tempo"}
            dic["Tempo"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_chords:
                dic["Tempo"] |= {"Chord"}
            if self.config.use_rests:
                dic["Tempo"].add("Rest")  # only for first token
            if self.config.use_pitch_intervals:
                dic["Tempo"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_time_signatures:
            dic["Bar"] = {"TimeSig"}
            if self.config.additional_params["use_bar_end_tokens"]:
                dic["Bar"].add("Bar")
            dic["TimeSig"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_chords:
                dic["TimeSig"] |= {"Chord"}
            if self.config.use_rests:
                dic["TimeSig"].add("Rest")  # only for first token
            if self.config.use_tempos:
                dic["Tempo"].add("TimeSig")
            if self.config.use_pitch_intervals:
                dic["TimeSig"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_sustain_pedals:
            dic["Position"].add("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = {"Duration"}
                if self.config.using_note_duration_tokens:
                    dic["Duration"].add("Pedal")
                elif self.config.use_velocities:
                    dic["Duration"] = {first_note_token_type, "Bar", "Position"}
                    dic["Velocity"].add("Pedal")
                else:
                    dic["Duration"] = {first_note_token_type, "Bar", "Position"}
                    dic["Pitch"].add("Pedal")
            else:
                dic["PedalOff"] = {
                    "Pedal",
                    "PedalOff",
                    first_note_token_type,
                    "Position",
                    "Bar",
                }
                dic["Pedal"] = {"Pedal", first_note_token_type, "Position", "Bar"}
                dic["Position"].add("PedalOff")
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
            if self.config.use_time_signatures:
                dic["TimeSig"].add("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["TimeSig"].add("PedalOff")
            if self.config.use_pitch_intervals:
                if self.config.sustain_pedal_duration:
                    dic["Duration"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                else:
                    dic["Pedal"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                    dic["PedalOff"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_pitch_bends:
            # As a Program token will precede PitchBend otherwise
            # Else no need to add Program as its already in
            dic["PitchBend"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_programs and not self.config.program_changes:
                dic["Program"].add("PitchBend")
            else:
                dic["Position"].add("PitchBend")
                if self.config.use_tempos:
                    dic["Tempo"].add("PitchBend")
                if self.config.use_time_signatures:
                    dic["TimeSig"].add("PitchBend")
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
            dic["Rest"] = {"Rest", first_note_token_type, "Position", "Bar"}
            dic[
                "Duration"
                if self.config.using_note_duration_tokens
                else "Velocity"
                if self.config.use_velocities
                else first_note_token_type
            ].add("Rest")
            if self.config.use_chords:
                dic["Rest"] |= {"Chord"}
            if self.config.use_tempos:
                dic["Rest"].add("Tempo")
            if self.config.use_time_signatures:
                dic["Rest"].add("TimeSig")
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
            for token_type in {
                "Position",
                "Rest",
                "PitchBend",
                "Pedal",
                "PedalOff",
                "Tempo",
                "TimeSig",
                "Chord",
            }:
                if token_type in dic:
                    dic["Program"].add(token_type)
                    dic[token_type].add("Program")

        if self.config.use_pitchdrum_tokens:
            dic["PitchDrum"] = dic["Pitch"]
            for key, values in dic.items():
                if "Pitch" in values:
                    dic[key].add("PitchDrum")

        return dic
