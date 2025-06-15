"""Compound Word tokenizer."""

from __future__ import annotations

import warnings

import numpy as np
from symusic import Note, Score, Tempo, TimeSignature, Track

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS, TIME_SIGNATURE
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat

_ADD_TOK_ATTRIBUTES = [
    "use_velocities",
    "using_note_duration_tokens",
    "use_programs",
    "use_chords",
    "use_rests",
    "use_tempos",
    "use_time_signatures",
]


class CPWord(MusicTokenizer):
    r"""
    Compound Word tokenizer.

    Introduced with the
    `Compound Word Transformer (Hsiao et al.) <https://ojs.aaai.org/index.php/AAAI/article/view/16091>`_,
    this tokenization is similar to :ref:`REMI` but uses embedding pooling operations
    to reduce the overall sequence length: note tokens (*Pitch*, *Velocity* and
    *Duration*) are first independently converted to embeddings which are then merged
    (pooled) into a single one.
    Each compound token will be a list of the form (index: Token type):

    * 0: Family;
    * 1: Bar/Position;
    * 2: Pitch;
    * (3: Velocity);
    * (4: Duration);
    * (+ Optional) Program: associated with notes (pitch/velocity/duration) or chords;
    * (+ Optional) Chord: chords occurring with position tokens;
    * (+ Optional) Rest: rest acting as a TimeShift token;
    * (+ Optional) Tempo: occurring with position tokens;
    * (+ Optional) TimeSig: occurring with bar tokens.

    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can
    be very delicate. Hence, we do not recommend this tokenization for generation with
    small models.
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when
    ``config.use_programs`` is False, only the tempos and time signatures of the first
    sequence will be decoded for the whole music.
    """

    def _tweak_config_before_creating_voc(self) -> None:
        if self.config.use_time_signatures and self.config.use_rests:
            # NOTE: this configuration could work by adding a Bar token with the new
            # TimeSig after the Rest, but the decoding should handle this to not add
            # another bar. Or it could work by making Rests not crossing new bars.
            # Rests would have a maximal value corresponding to the difference between
            # the previous event tick and the tick of the next bar. However, in cases
            # of long rests of more than one bar, we would have successions of
            # Rest --> Bar --> Rest --> Bar ... tokens.
            warnings.warn(
                "You are using both Time Signatures and Rests with CPWord. Be aware"
                "that this configuration can result in altered time, as the time"
                "signature is carried by the Bar tokens, that are skipped during"
                "rests. To disable this warning, you can disable either Time"
                "Signatures or Rests. Otherwise, you can check that your data does"
                "not have time signature changes occurring during rests.",
                stacklevel=2,
            )

        # Durations are enabled for all programs or none
        if any(
            p not in self.config.use_note_duration_programs
            for p in self.config.programs
        ):
            self.config.use_note_duration_programs = self.config.programs
            warnings.warn(
                "Setting note duration programs to `tokenizer.config.programs`."
                "CPWord only allows to use note duration tokens for either all "
                "programs or none.",
                stacklevel=2,
            )

        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.program_changes = False
        self._disable_attribute_controls()
        token_types = ["Family", "Position", "Pitch"]
        for add_tok_attr, add_token in [
            ("use_velocities", "Velocity"),
            ("using_note_duration_tokens", "Duration"),
            ("use_programs", "Program"),
            ("use_chords", "Chord"),
            ("use_rests", "Rest"),
            ("use_tempos", "Tempo"),
            ("use_time_signatures", "TimeSig"),
        ]:
            if getattr(self.config, add_tok_attr):
                token_types.append(add_token)
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation
        self.vocab_types_idx["Bar"] = 1  # same as position

    def _add_time_events(
        self, events: list[Event], time_division: int
    ) -> list[list[Event]]:
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
        duration_offset = 0
        if self.config.use_velocities:
            duration_offset += 1
        if self.config.using_note_duration_tokens:
            duration_offset += 1
        all_events = []
        current_bar = -1
        bar_at_last_ts_change = 0
        previous_tick = -1
        previous_note_end = 0
        tick_at_last_ts_change = tick_at_current_bar = 0
        current_time_sig = TIME_SIGNATURE
        if self.config.log_tempos:
            # pick the closest to the default value
            current_tempo = float(
                self.tempos[(np.abs(self.tempos - self.default_tempo)).argmin()]
            )
        else:
            current_tempo = self.default_tempo
        current_program = None
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(0, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
        # First look for a TimeSig token, if any is given at tick 0, to update
        # current_time_sig
        if self.config.use_time_signatures:
            for event in events:
                # There should be a TimeSig token at tick 0
                if event.type_ == "TimeSig":
                    current_time_sig = list(map(int, event.value.split("/")))
                    ticks_per_bar = compute_ticks_per_bar(
                        TimeSignature(event.time, *current_time_sig), time_division
                    )
                    ticks_per_beat = compute_ticks_per_beat(
                        current_time_sig[1], time_division
                    )
                    ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
                    break
        # Then look for a Tempo token, if any is given at tick 0, to update
        # current_tempo
        if self.config.use_tempos:
            for event in events:
                if event.type_ == "Tempo":
                    current_tempo = event.value
                    break
                if event.type_ in {
                    "Pitch",
                    "PitchDrum",
                    "Velocity",
                    "Duration",
                    "PitchBend",
                    "Pedal",
                }:
                    break
        # Add the time events
        for e, event in enumerate(events):
            if event.type_ == "Tempo":
                current_tempo = event.value
            elif event.type_ == "Program":
                current_program = event.value
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
                            self.__create_cp_token(
                                previous_tick,
                                rest=".".join(map(str, dur_value)),
                                desc=f"{event.time - previous_tick} ticks",
                            )
                        )
                        previous_tick += dur_ticks
                    # We update current_bar and tick_at_current_bar here without
                    # creating Bar tokens
                    real_current_bar = (
                        bar_at_last_ts_change
                        + (previous_tick - tick_at_last_ts_change) // ticks_per_bar
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
                num_new_bars = (
                    bar_at_last_ts_change
                    + (event.time - tick_at_last_ts_change) // ticks_per_bar
                    - current_bar
                )
                if num_new_bars >= 1:
                    if self.config.use_time_signatures:
                        time_sig_arg = f"{current_time_sig[0]}/{current_time_sig[1]}"
                    else:
                        time_sig_arg = None
                    for i in range(num_new_bars):
                        # exception when last bar and event.type == "TimeSig"
                        if i == num_new_bars - 1 and event.type_ == "TimeSig":
                            time_sig_arg = list(map(int, event.value.split("/")))
                            time_sig_arg = f"{time_sig_arg[0]}/{time_sig_arg[1]}"
                        all_events.append(
                            self.__create_cp_token(
                                (current_bar + i + 1) * ticks_per_bar,
                                bar=True,
                                desc="Bar",
                                time_signature=time_sig_arg,
                            )
                        )
                    current_bar += num_new_bars
                    tick_at_current_bar = (
                        tick_at_last_ts_change
                        + (current_bar - bar_at_last_ts_change) * ticks_per_bar
                    )

                # Position
                if event.type_ != "TimeSig":
                    pos_index = (event.time - tick_at_current_bar) // ticks_per_pos
                    all_events.append(
                        self.__create_cp_token(
                            event.time,
                            pos=pos_index,
                            chord=event.value if event.type_ == "Chord" else None,
                            tempo=current_tempo if self.config.use_tempos else None,
                            desc="Position",
                        )
                    )

                previous_tick = event.time

            # Update time signature time variables, after adjusting the time (above)
            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                bar_at_last_ts_change += (
                    event.time - tick_at_last_ts_change
                ) // ticks_per_bar
                tick_at_last_ts_change = event.time
                ticks_per_bar = compute_ticks_per_bar(
                    TimeSignature(event.time, *current_time_sig), time_division
                )
                ticks_per_beat = compute_ticks_per_beat(
                    current_time_sig[1], time_division
                )
                ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
                # We decrease the previous tick so that a Position token is enforced
                # for the next event
                previous_tick -= 1

            # Convert event to CP Event
            # Update max offset time of the notes encountered
            if event.type_ in {"Pitch", "PitchDrum"} and e + duration_offset < len(
                events
            ):
                all_events.append(
                    self.__create_cp_token(
                        event.time,
                        pitch=event.value,
                        vel=events[e + 1].value if self.config.use_velocities else None,
                        dur=events[e + duration_offset].value
                        if self.config.using_note_duration_tokens
                        else None,
                        program=current_program,
                        pitch_drum=event.type_ == "PitchDrum",
                    )
                )
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type_ in [
                "Program",
                "Tempo",
                "TimeSig",
                "Chord",
            ]:
                previous_note_end = max(previous_note_end, event.time)

        return all_events

    def __create_cp_token(
        self,
        time: int,
        bar: bool = False,
        pos: int | None = None,
        pitch: int | None = None,
        vel: int | None = None,
        dur: str | None = None,
        chord: str | None = None,
        rest: str | None = None,
        tempo: float | None = None,
        time_signature: str | None = None,
        program: int | None = None,
        desc: str = "",
        pitch_drum: bool = False,
    ) -> list[Event]:
        r"""
        Create a CP Word token.

        It follows the structure:
            (index. Token type)
            0. *Family*
            1. *Bar*/*Position*
            2. *Pitch*
            (3. *Velocity*)
            (4. *Duration*)
            (5. *Program*) optional, with notes (pitch/velocity/duration) or chords
            (6. *Chord*) optional, chords occurring with position tokens
            (7. *Rest*) optional, rest acting as a TimeShift token
            (8. *Tempo*) optional, occurring with position tokens
            (9. *TimeSig*) optional, occurring with bar tokens
        **Note**: the first Family token (first in list) will be given as an ``Event``
        object to keep track of time easily so that other method can sort CP tokens
        afterward.

        :param time: the current tick
        :param bar: True if this token represents a new bar occurring
        :param pos: the position index
        :param pitch: note pitch
        :param vel: note velocity
        :param dur: note duration
        :param chord: chord value
        :param rest: rest value
        :param tempo: tempo index
        :param program: a program number if you want to produce a Program CP token
            (read note above)
        :param desc: an optional argument for debug and used to spot position tokens
            in track_to_tokens
        :param pitch_drum: will create a ``PitchDrum`` token instead of ``Pitch``.
        :return: The compound token as a list of integers
        """

        def create_event(type_: str, value: str | int) -> Event:
            return Event(type_=type_, value=value, time=time, desc=desc)

        cp_token = [
            Event(type_="Family", value="Metric", time=time, desc=desc),
            Event(type_="Ignore", value="None", time=time, desc=desc),
            Event(type_="Ignore", value="None", time=time, desc=desc),
        ]
        cp_token += [
            create_event("Ignore", "None")
            for add_tok_attr in _ADD_TOK_ATTRIBUTES
            if getattr(self.config, add_tok_attr)
        ]

        if bar:
            cp_token[1] = create_event("Bar", "None")
            if time_signature is not None:
                cp_token[self.vocab_types_idx["TimeSig"]] = create_event(
                    "TimeSig", time_signature
                )
        elif pos is not None:
            cp_token[1] = create_event("Position", pos)
            if chord is not None:
                cp_token[self.vocab_types_idx["Chord"]] = create_event("Chord", chord)
            if tempo is not None:
                cp_token[self.vocab_types_idx["Tempo"]] = create_event(
                    "Tempo", str(tempo)
                )
        elif rest is not None:
            cp_token[self.vocab_types_idx["Rest"]] = create_event("Rest", rest)
        elif pitch is not None:
            pitch_token_name = "PitchDrum" if pitch_drum else "Pitch"
            cp_token[0].value = "Note"
            cp_token[2] = create_event(pitch_token_name, pitch)
            if self.config.use_velocities:
                cp_token[3] = create_event("Velocity", vel)  # shouldn't be None
            if dur:
                cp_token[self.vocab_types_idx["Duration"]] = create_event(
                    "Duration", dur
                )
            if program is not None:
                cp_token[self.vocab_types_idx["Program"]] = create_event(
                    "Program", program
                )

        return cp_token

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
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        score = Score(self.time_division)

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes = [Tempo(-1, self.default_tempo)]
        time_signature_changes = []
        tempo_changes[0].tempo = -1

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

        current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
        current_bar = -1
        bar_at_last_ts_change = 0
        current_program = 0
        current_track = None
        previous_note_end = 0
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0:
                if self.config.use_time_signatures:
                    for compound_token in seq:
                        token_family = compound_token[0].split("_")[1]
                        if token_family == "Metric":
                            bar_pos = compound_token[1].split("_")[0]
                            if bar_pos == "Bar":
                                num, den = self._parse_token_time_signature(
                                    compound_token[
                                        self.vocab_types_idx["TimeSig"]
                                    ].split("_")[1]
                                )
                                time_signature_changes.append(
                                    TimeSignature(0, num, den)
                                )
                                break
                        else:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[0]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, score.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
                current_bar = -1
                bar_at_last_ts_change = 0
                previous_note_end = 0
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs:
                    for compound_token in seq:
                        if compound_token[0].split("_")[1] == "Note":
                            current_program = int(
                                compound_token[self.vocab_types_idx["Program"]].split(
                                    "_"
                                )[1]
                            )
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

            # Decode tokens
            for compound_token in seq:
                token_family = compound_token[0].split("_")[1]
                if token_family == "Note":
                    pad_range_idx = 3
                    if self.config.use_velocities:
                        pad_range_idx += 1
                    if self.config.using_note_duration_tokens:
                        pad_range_idx += 1
                    if self.config.use_programs:
                        pad_range_idx += 1
                    if any(
                        tok.split("_")[1] == "None"
                        for tok in compound_token[2:pad_range_idx]
                    ):
                        continue
                    pitch = int(compound_token[2].split("_")[1])
                    vel = (
                        int(compound_token[3].split("_")[1])
                        if self.config.use_velocities
                        else DEFAULT_VELOCITY
                    )
                    if self.config.using_note_duration_tokens:
                        duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                            compound_token[self.vocab_types_idx["Duration"]].split("_")[
                                1
                            ]
                        ]
                    else:
                        duration = int(
                            self.config.default_note_duration * ticks_per_beat
                        )
                    new_note = Note(current_tick, duration, pitch, vel)
                    if (
                        self.config.one_token_stream_for_programs
                        and self.config.use_programs
                    ):
                        current_program = int(
                            compound_token[self.vocab_types_idx["Program"]].split("_")[
                                1
                            ]
                        )
                        check_inst(current_program)
                        tracks[current_program].notes.append(new_note)
                    else:
                        current_track.notes.append(new_note)
                    previous_note_end = max(previous_note_end, current_tick + duration)

                elif token_family == "Metric":
                    bar_pos = compound_token[1].split("_")[0]
                    if bar_pos == "Bar":
                        current_bar += 1
                        if current_bar > 0:
                            current_tick = tick_at_current_bar + ticks_per_bar
                        tick_at_current_bar = current_tick
                        # Add new TS only if different from the last one
                        if self.config.use_time_signatures:
                            num, den = self._parse_token_time_signature(
                                compound_token[self.vocab_types_idx["TimeSig"]].split(
                                    "_"
                                )[1]
                            )
                            if (
                                num != current_time_sig.numerator
                                or den != current_time_sig.denominator
                            ):
                                current_time_sig = TimeSignature(current_tick, num, den)
                                if si == 0:
                                    time_signature_changes.append(current_time_sig)
                                tick_at_last_ts_change = tick_at_current_bar
                                bar_at_last_ts_change = current_bar
                                ticks_per_bar = compute_ticks_per_bar(
                                    current_time_sig, score.ticks_per_quarter
                                )
                                ticks_per_beat = self._tpb_per_ts[
                                    current_time_sig.denominator
                                ]
                                ticks_per_pos = (
                                    ticks_per_beat // self.config.max_num_pos_per_beat
                                )
                    elif bar_pos == "Position":  # i.e. its a position
                        if current_bar == -1:
                            # in case this Position token comes before any Bar token
                            current_bar = 0
                        current_tick = (
                            tick_at_current_bar
                            + int(compound_token[1].split("_")[1]) * ticks_per_pos
                        )
                        # Add new tempo change only if different from the last one
                        if self.config.use_tempos and si == 0:
                            tempo = float(
                                compound_token[self.vocab_types_idx["Tempo"]].split(
                                    "_"
                                )[1]
                            )
                            if (
                                tempo != round(tempo_changes[-1].tempo, 2)
                                and current_tick != tempo_changes[-1].time
                            ):
                                tempo_changes.append(Tempo(current_tick, tempo))
                    elif (
                        self.config.use_rests
                        and compound_token[self.vocab_types_idx["Rest"]].split("_")[1]
                        != "None"
                    ):
                        current_tick = max(previous_note_end, current_tick)
                        current_tick += self._tpb_rests_to_ticks[ticks_per_beat][
                            compound_token[self.vocab_types_idx["Rest"]].split("_")[1]
                        ]
                        real_current_bar = (
                            bar_at_last_ts_change
                            + (current_tick - tick_at_last_ts_change) // ticks_per_bar
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

                    previous_note_end = max(previous_note_end, current_tick)

            # Add current_inst to score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        # Delete mocked
        # And handle first tempo (tick 0) here instead of super
        del tempo_changes[0]
        if len(tempo_changes) == 0 or (
            tempo_changes[0].time != 0
            and round(tempo_changes[0].tempo, 2) != self.default_tempo
        ):
            tempo_changes.insert(0, Tempo(0, self.default_tempo))
        elif round(tempo_changes[0].tempo, 2) == self.default_tempo:
            tempo_changes[0].time = 0

        # Add global events to score
        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes

        return score

    def _create_base_vocabulary(self) -> list[list[str]]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = [["Family_Metric", "Family_Note"]]

        # POSITION
        # self.time_division is equal to the maximum possible ticks/beat value.
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab.append(
            [
                "Ignore_None",
                "Bar_None",
                *[f"Position_{i}" for i in range(num_positions)],
            ]
        )

        # PITCH
        vocab.append(
            [
                "Ignore_None",
                *[
                    f"Pitch_{i}"
                    for i in range(
                        self.config.pitch_range[0],
                        self.config.pitch_range[1] + 1,
                    )
                ],
            ]
        )
        if self.config.use_pitchdrum_tokens:
            vocab[2] += [
                f"PitchDrum_{i}"
                for i in range(
                    self.config.drums_pitch_range[0],
                    self.config.drums_pitch_range[1] + 1,
                )
            ]

        # VELOCITY
        if self.config.use_velocities:
            vocab.append(["Ignore_None", *[f"Velocity_{i}" for i in self.velocities]])

        # DURATION
        if self.config.using_note_duration_tokens:
            vocab.append(
                [
                    "Ignore_None",
                    *[
                        f"Duration_{'.'.join(map(str, duration))}"
                        for duration in self.durations
                    ],
                ]
            )

        # PROGRAM
        if self.config.use_programs:
            vocab += [
                ["Ignore_None"]
                + [f"Program_{program}" for program in self.config.programs]
            ]

        # CHORD
        if self.config.use_chords:
            vocab.append(["Ignore_None", *self._create_chords_tokens()])

        # REST
        if self.config.use_rests:
            vocab += [
                ["Ignore_None"]
                + [f"Rest_{'.'.join(map(str, rest))}" for rest in self.rests]
            ]

        # TEMPO
        if self.config.use_tempos:
            vocab += [["Ignore_None"] + [f"Tempo_{i}" for i in self.tempos]]

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            vocab += [
                ["Ignore_None"]
                + [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]
            ]

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = {
            "Bar": {"Position", "Bar"},
            "Position": {"Pitch"},
            "Pitch": {"Pitch", "Bar", "Position"},
        }

        if self.config.use_chords:
            dic["Rest"] = {"Rest", "Position"}
            dic["Pitch"] |= {"Rest"}

        if self.config.use_rests:
            dic["Rest"] = {"Rest", "Position", "Bar"}
            dic["Pitch"] |= {"Rest"}

        if self.config.use_tempos:
            # Because a tempo change can happen at any moment
            dic["Position"] |= {"Position", "Bar"}
            if self.config.use_rests:
                dic["Position"].add("Rest")
                dic["Rest"].add("Position")

        for key in dic:
            dic[key].add("Ignore")
        dic["Ignore"] = set(dic.keys())

        if self.config.use_pitchdrum_tokens:
            dic["PitchDrum"] = dic["Pitch"]
            for key, values in dic.items():
                if "Pitch" in values:
                    dic[key].add("PitchDrum")

        return dic

    def _tokens_errors(self, tokens: list[list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """

        def cp_token_type(tok: list[str]) -> list[str]:
            family = tok[0].split("_")[1]
            msg_err = "No token type found, unknown error"
            if family == "Note":
                return tok[2].split("_")
            if family == "Metric":
                bar_pos = tok[1].split("_")
                if bar_pos[0] in ["Bar", "Position"]:
                    return bar_pos
                # additional token
                for i in range(1, 5):
                    decoded_token = tok[-i].split("_")
                    if decoded_token[0] != "Ignore":
                        return decoded_token
                raise RuntimeError(msg_err)
            if family == "None":
                return ["PAD", "None"]
            raise RuntimeError(msg_err)

        err = 0
        previous_type = cp_token_type(tokens[0])[0]
        current_pos = -1
        program = 0
        current_pitches = {p: [] for p in self.config.programs}

        for token in tokens[1:]:
            token_type, token_value = cp_token_type(token)
            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Bar":
                    current_pos = -1
                    current_pitches = {p: [] for p in self.config.programs}
                elif self.config.remove_duplicated_notes and token_type in {
                    "Pitch",
                    "PitchDrum",
                }:
                    if self.config.use_programs:
                        program = int(self[5, token[5]].split("_")[1])
                    if int(token_value) in current_pitches[program]:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches[program].append(int(token_value))
                elif token_type == "Position":
                    if int(token_value) <= current_pos and previous_type != "Rest":
                        err += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = {p: [] for p in self.config.programs}
            # Bad token type
            else:
                err += 1
            previous_type = token_type

        return err
