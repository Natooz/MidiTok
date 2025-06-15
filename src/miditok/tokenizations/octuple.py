"""Octuple tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from symusic import Note, Score, Tempo, TimeSignature, Track

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS, TIME_SIGNATURE
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat, get_bars_ticks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class Octuple(MusicTokenizer):
    r"""
    Octuple tokenizer.

    Introduced with `MusicBert (Zeng et al.) <https://arxiv.org/abs/2106.05630>`_,
    the idea of Octuple is to use embedding pooling so that each pooled embedding
    represents a single note. Tokens (*Pitch*, *Velocity*...) are first independently
    converted to embeddings which are then merged (pooled) into a single one.
    Each pooled token will be a list of the form (index: Token type):

    * 0: Pitch/PitchDrum;
    * 1: Position;
    * 2: Bar;
    * (+ Optional) Velocity;
    * (+ Optional) Duration;
    * (+ Optional) Program;
    * (+ Optional) Tempo;
    * (+ Optional) TimeSignature.

    Its considerably reduces the sequence lengths, while handling multitrack.
    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can
    be very delicate. Hence, we do not recommend this tokenization for generation with
    small models.

    **Notes:**

    * As the time signature is carried simultaneously with the note tokens, if a Time
        Signature change occurs and that the following bar do not contain any note, the
        time will be shifted by one or multiple bars depending on the previous time
        signature numerator and time gap between the last and current note. Octuple
        cannot represent time signature accurately, hence some unavoidable errors of
        conversion can happen. **For this reason, Octuple is implemented with Time
        Signature but tested without.**
    * Tokens are first sorted by time, then track, then pitch values.
    * Tracks with the same *Program* will be merged.
    * When decoding multiple token sequences (of multiple tracks), i.e. when
        `config.use_programs` is False, only the tempos and time signatures of the
        first sequence will be decoded for the whole music.
    """

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.delete_equal_successive_tempo_changes = True
        self.config.program_changes = False
        self._disable_attribute_controls()

        # Durations are enabled for all programs or none
        if any(
            p not in self.config.use_note_duration_programs
            for p in self.config.programs
        ):
            self.config.use_note_duration_programs = self.config.programs
            warn(
                "Setting note duration programs to `tokenizer.config.programs`."
                "Octuple only allows to use note duration tokens for either all "
                "programs or none.",
                stacklevel=2,
            )

        # used in place of positional encoding
        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = 60

        token_types = ["Pitch", "Position", "Bar"]
        if self.config.use_velocities:
            token_types.append("Velocity")
        if self.config.using_note_duration_tokens:
            token_types.append("Duration")
        if self.config.use_programs:
            token_types.append("Program")
        if self.config.use_tempos:
            token_types.append("Tempo")
        if self.config.use_time_signatures:
            token_types.append("TimeSig")
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation

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
        current_bar = 0
        current_bar_from_ts_time = 0
        current_tick_from_ts_time = 0
        current_pos = 0
        previous_tick = 0
        current_time_sig = TIME_SIGNATURE
        current_tempo = self.default_tempo
        current_program = 0
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(0, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
        for e, event in enumerate(events):
            # Set current bar and position
            # This is done first, as we need to compute these values with the current
            # ticks_per_bar, which might change if the current event is a TimeSig
            if event.time != previous_tick:
                elapsed_tick = event.time - current_tick_from_ts_time
                current_bar = current_bar_from_ts_time + elapsed_tick // ticks_per_bar
                tick_at_current_bar = (
                    current_tick_from_ts_time
                    + (current_bar - current_bar_from_ts_time) * ticks_per_bar
                )
                current_pos = (event.time - tick_at_current_bar) // ticks_per_pos
                previous_tick = event.time

            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                current_bar_from_ts_time = current_bar
                current_tick_from_ts_time = previous_tick
                ticks_per_bar = compute_ticks_per_bar(
                    TimeSignature(event.time, *current_time_sig), time_division
                )
                ticks_per_beat = compute_ticks_per_beat(
                    current_time_sig[1], time_division
                )
                ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            elif event.type_ == "Tempo":
                current_tempo = event.value
            elif event.type_ == "Program":
                current_program = event.value
            elif event.type_ in {"Pitch", "PitchDrum"} and e + duration_offset < len(
                events
            ):
                pitch_token_name = (
                    "PitchDrum" if event.type_ == "PitchDrum" else "Pitch"
                )
                new_event = [
                    Event(type_=pitch_token_name, value=event.value, time=event.time),
                    Event(type_="Position", value=current_pos, time=event.time),
                    Event(type_="Bar", value=current_bar, time=event.time),
                ]
                if self.config.use_velocities:
                    new_event.append(
                        Event(
                            type_="Velocity", value=events[e + 1].value, time=event.time
                        ),
                    )
                if self.config.using_note_duration_tokens:
                    new_event.append(
                        Event(
                            type_="Duration",
                            value=events[e + duration_offset].value,
                            time=event.time,
                        )
                    )
                if self.config.use_programs:
                    new_event.append(Event("Program", current_program))
                if self.config.use_tempos:
                    new_event.append(Event(type_="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type_="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
                all_events.append(new_event)

        return all_events

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        We override the parent method in order to check the number of bars in the file.

        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If
        ``config.one_token_stream_for_programs`` is enabled, all events of all tracks
        are treated all at once, otherwise the events of each track are treated
        independently.

        :param score: the :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        del attribute_controls_indexes
        # Check bar embedding limit, update if needed
        bar_ticks = get_bars_ticks(score, only_notes_onsets=True)
        if self.config.additional_params["max_bar_embedding"] < len(bar_ticks):
            score = score.clip(
                0, bar_ticks[self.config.additional_params["max_bar_embedding"]]
            )
            msg = (
                f"miditok: {type(self).__name__} cannot tokenize entirely this file "
                f"as it contains {len(bar_ticks)} bars whereas the limit of the "
                f"tokenizer is {self.config.additional_params['max_bar_embedding']}. "
                "It is therefore clipped to "
                f"{self.config.additional_params['max_bar_embedding']} bars."
            )
            warn(msg, stacklevel=2)

        return super()._score_to_tokens(score)

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
        tempo_changes, time_signature_changes = [Tempo(-1, self.default_tempo)], []
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

        bar_at_last_ts_change = 0
        tick_at_last_ts_change = 0
        current_program = 0
        current_track = None
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0 and self.config.use_time_signatures:
                num, den = self._parse_token_time_signature(
                    seq[0][self.vocab_types_idx["TimeSig"]].split("_")[1]
                )
                time_signature_changes.append(TimeSignature(0, num, den))
            else:
                time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[0]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, score.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs and len(seq) > 0:
                    current_program = int(
                        seq[0][self.vocab_types_idx["Program"]].split("_")[1]
                    )
                    if current_program == -1:
                        is_drum, current_program = True, 0
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )

            # Decode tokens
            for time_step in seq:
                num_tok_to_check = 3
                for attr in ("use_velocities", "use_programs"):
                    if getattr(self.config, attr):
                        num_tok_to_check += 1
                if self.config.use_programs:
                    current_program = int(
                        time_step[self.vocab_types_idx["Program"]].split("_")[1]
                    )

                if self.config.using_note_duration_tokens:
                    num_tok_to_check += 1
                if any(
                    tok.split("_")[1] == "None" for tok in time_step[:num_tok_to_check]
                ):
                    # Padding or mask: error of prediction or end of sequence anyway
                    continue

                # Note attributes
                pitch = int(time_step[0].split("_")[1])
                vel = (
                    int(time_step[self.vocab_types_idx["Velocity"]].split("_")[1])
                    if self.config.use_velocities
                    else DEFAULT_VELOCITY
                )

                # Time values
                event_pos = int(time_step[1].split("_")[1])
                event_bar = int(time_step[2].split("_")[1])
                current_tick = (
                    tick_at_last_ts_change
                    + (event_bar - bar_at_last_ts_change) * ticks_per_bar
                    + event_pos * ticks_per_pos
                )

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
                        num != current_time_sig.numerator
                        or den != current_time_sig.denominator
                    ):
                        # tick from bar of ts change
                        tick_at_last_ts_change += (
                            event_bar - bar_at_last_ts_change
                        ) * ticks_per_bar
                        current_time_sig = TimeSignature(
                            tick_at_last_ts_change, num, den
                        )
                        if si == 0:
                            time_signature_changes.append(current_time_sig)
                        bar_at_last_ts_change = event_bar
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, score.ticks_per_quarter
                        )
                        ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
                        ticks_per_pos = (
                            ticks_per_beat // self.config.max_num_pos_per_beat
                        )

                # Note duration
                if self.config.using_note_duration_tokens:
                    duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                        time_step[self.vocab_types_idx["Duration"]].split("_")[1]
                    ]
                else:
                    duration = int(self.config.default_note_duration * ticks_per_beat)

                # Append the created note
                new_note = Note(current_tick, duration, pitch, vel)
                if self.config.one_token_stream_for_programs:
                    check_inst(current_program)
                    tracks[current_program].notes.append(new_note)
                else:
                    current_track.notes.append(new_note)

                # Tempo, adds a TempoChange if necessary
                if (
                    si == 0
                    and self.config.use_tempos
                    and time_step[self.vocab_types_idx["Tempo"]].split("_")[1] != "None"
                ):
                    tempo = float(
                        time_step[self.vocab_types_idx["Tempo"]].split("_")[1]
                    )
                    if tempo != round(tempo_changes[-1].tempo, 2):
                        tempo_changes.append(Tempo(current_tick, tempo))

            # Add current_inst to the score and handle notes still active
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

        # Add global events to the score
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
        # PITCH
        vocab = [
            [
                f"Pitch_{i}"
                for i in range(
                    self.config.pitch_range[0], self.config.pitch_range[1] + 1
                )
            ]
        ]
        if self.config.use_pitchdrum_tokens:
            vocab[0] += [
                f"PitchDrum_{i}"
                for i in range(
                    self.config.drums_pitch_range[0],
                    self.config.drums_pitch_range[1] + 1,
                )
            ]

        # POSITION
        # self.time_division is equal to the maximum possible ticks/beat value.
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab.append([f"Position_{i}" for i in range(num_positions)])

        # BAR (positional encoding)
        vocab.append(
            [
                f"Bar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
        )

        # OPTIONAL TOKENS STARTING HERE

        # VELOCITY
        if self.config.use_velocities:
            vocab.append([f"Velocity_{i}" for i in self.velocities])

        # DURATION
        if self.config.using_note_duration_tokens:
            vocab.append(
                [
                    f"Duration_{'.'.join(map(str, duration))}"
                    for duration in self.durations
                ]
            )

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

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        Not relevant for Octuple as it is not subject to token type errors.

        :return: the token types transitions dictionary.
        """
        return {}

    def _tokens_errors(self, tokens: list[list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        The token types are always the same in Octuple so this method only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in
                time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played
                at the current position.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err = 0
        current_bar = current_pos = -1
        current_pitches = {p: [] for p in self.config.programs}
        current_program = 0

        for token in tokens:
            if any(tok.split("_")[1] == "None" for tok in token):
                err += 1
                continue
            has_error = False
            bar_value = int(token[2].split("_")[1])
            pos_value = int(token[1].split("_")[1])
            pitch_value = int(token[0].split("_")[1])
            if self.config.use_programs:
                current_program = int(
                    token[self.vocab_types_idx["Program"]].split("_")[1]
                )

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
            if self.config.remove_duplicated_notes:
                if pitch_value in current_pitches[current_program]:
                    has_error = True
                else:
                    current_pitches[current_program].append(pitch_value)

            if has_error:
                err += 1

        return err
