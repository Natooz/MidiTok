"""Structured tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from symusic import Note, Score, Track

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils.utils import np_get_closest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class Structured(MusicTokenizer):
    r"""
    Structured tokenizer, with a recurrent token type succession.

    Introduced with the `Piano Inpainting Application <https://arxiv.org/abs/2002.00212>`_,
    it is similar to :ref:`TSD` but is based on a consistent token type successions.
    Token types always follow the same pattern: *Pitch* -> *Velocity* -> *Duration* ->
    *TimeShift*. The latter is set to 0 for simultaneous notes. To keep this property,
    no additional token can be inserted in MidiTok's implementation, except *Program*
    that can optionally be added preceding ``Pitch`` tokens. If you specify
    ``use_programs`` as ``True`` in the config file, the tokenizer will add *Program*
    tokens before each *Pitch* tokens to specify its instrument, and will treat all
    tracks as a single stream of tokens.

    **Note:** as ``Structured`` uses *TimeShifts* events to move the time from note to
    note, it can be unsuited for tracks with pauses longer than the maximum *TimeShift*
    value. In such cases, the maximum *TimeShift* value will be used.
    """

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_tempos = False
        self.config.use_time_signatures = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.program_changes = False
        self._disable_attribute_controls()

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: np.ndarray | None = None,
        time_division: int | None = None,
        ticks_bars: Sequence[int] | None = None,
        ticks_beats: Sequence[int] | None = None,
        add_track_attribute_controls: bool = False,
        bar_idx_attribute_controls: Sequence[int] | None = None,
    ) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in** ``preprocess_score``.

        :param track: ``symusic.Track`` to extract events from.
        :return: sequence of corresponding ``Event``s.
        """
        del (
            time_division,
            ticks_bars,
            ticks_beats,
            add_track_attribute_controls,
            bar_idx_attribute_controls,
        )
        # Make sure the notes are sorted first by their onset (start) times, second by
        # pitch: notes.sort(key=lambda x: (x.start, x.pitch)) done in preprocess_score
        program = track.program if not track.is_drum else -1
        use_durations = program in self.config.use_note_duration_programs
        events = []

        # Creates the Note On, Note Off and Velocity events
        previous_tick = 0
        ticks_per_beat = self.time_division
        for note in track.notes:
            # In this case, we directly add TimeShift events here so we don't have to
            # call __add_time_note_events and avoid delay cause by event sorting
            if not self.config.one_token_stream_for_programs:
                time_shift_ticks = note.start - previous_tick
                if time_shift_ticks != 0:
                    time_shift_ticks = int(
                        np_get_closest(
                            self._tpb_to_time_array[ticks_per_beat],
                            np.array([time_shift_ticks]),
                        )
                    )
                    time_shift = self._tpb_ticks_to_tokens[ticks_per_beat][
                        time_shift_ticks
                    ]
                else:
                    time_shift = "0.0.1"
                events.append(
                    Event(
                        type_="TimeShift",
                        time=note.start,
                        desc=f"{time_shift_ticks} ticks",
                        value=time_shift,
                    )
                )
            # Note On / Velocity / Duration
            if self.config.use_programs:
                events.append(
                    Event(
                        type_="Program", value=program, time=note.start, desc=note.end
                    )
                )
            pitch_token_name = (
                "PitchDrum"
                if track.is_drum and self.config.use_pitchdrum_tokens
                else "Pitch"
            )
            events.append(
                Event(
                    type_=pitch_token_name,
                    value=note.pitch,
                    time=note.start,
                    desc=note.end,
                )
            )
            if self.config.use_velocities:
                events.append(
                    Event(
                        type_="Velocity",
                        value=note.velocity,
                        time=note.start,
                        desc=f"{note.velocity}",
                    )
                )
            if use_durations:
                dur = self._tpb_ticks_to_tokens[ticks_per_beat][note.duration]
                events.append(
                    Event(
                        type_="Duration",
                        value=dur,
                        time=note.start,
                        desc=f"{note.duration} ticks",
                    )
                )
            previous_tick = note.start

        return events

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
        token_types_to_check = (
            {"Program"}
            if self.config.one_token_stream_for_programs
            else {"Pitch", "PitchDrum"}
        )

        # Add "TimeShift" tokens before each "Pitch" tokens
        previous_tick = 0
        for event in events:
            if event.type_ in token_types_to_check:
                # Time shift
                time_shift_ticks = event.time - previous_tick
                if time_shift_ticks != 0:
                    time_shift_ticks = int(
                        np_get_closest(
                            self._tpb_to_time_array[time_division],
                            np.array([time_shift_ticks]),
                        )
                    )
                    time_shift = self._tpb_ticks_to_tokens[time_division][
                        time_shift_ticks
                    ]
                else:
                    time_shift = "0.0.1"
                all_events.append(
                    Event(
                        type_="TimeShift",
                        time=event.time,
                        desc=f"{time_shift_ticks} ticks",
                        value=time_shift,
                    )
                )
                previous_tick = event.time

            all_events.append(event)

        return all_events

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        We override the parent method to handle the "non-program" case where
        *TimeShift* events have already been added by ``_notes_to_events``.

        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If
        ``config.one_token_stream_for_programs` is enabled, all events of all tracks
        are treated all at once, otherwise the events of each track are treated
        independently.

        :param score: the :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        del attribute_controls_indexes
        # Convert each track to tokens
        all_events = []

        # Adds note tokens
        if not self.config.one_token_stream_for_programs and len(score.tracks) == 0:
            all_events.append([])
        for track in score.tracks:
            note_events = self._create_track_events(track)
            if self.config.one_token_stream_for_programs:
                all_events += note_events
            else:
                all_events.append(note_events)

        # Add time events
        if self.config.one_token_stream_for_programs:
            if len(score.tracks) > 1:
                all_events.sort(key=lambda x: x.time)
            all_events = self._add_time_events(all_events, score.ticks_per_quarter)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                # No call to __add_time_note_events here as not needed
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

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
        dur_offset = 2 if self.config.use_velocities else 1

        # RESULTS
        instruments: dict[int, Track] = {}

        def check_inst(prog: int) -> None:
            if prog not in instruments:
                instruments[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        current_tick = 0
        current_program = 0
        current_track = None
        ticks_per_beat = score.ticks_per_quarter
        for si, seq in enumerate(tokens):
            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                current_tick = 0
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
                token_type, token_val = token.split("_")
                if token_type == "TimeShift" and token_val != "0.0.1":
                    current_tick += self._tpb_tokens_to_ticks[ticks_per_beat][token_val]
                elif token_type in {"Pitch", "PitchDrum"}:
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
                            pitch = int(seq[ti].split("_")[1])
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            new_note = Note(current_tick, dur, pitch, int(vel))
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                instruments[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                    except IndexError:
                        # A well constituted sequence should not raise an exception,
                        # however with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass
                elif token_type == "Program":
                    current_program = int(token_val)
                    current_track_use_duration = (
                        current_program in self.config.use_note_duration_programs
                    )

            # Add current_inst to score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        if self.config.one_token_stream_for_programs:
            score.tracks = list(instruments.values())

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

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # NoteOn/NoteOff/Velocity
        self._add_note_tokens_to_vocab_list(vocab)

        # TimeShift (same as durations)
        vocab.append("TimeShift_0.0.1")  # for a time shift of 0
        vocab += [
            f"TimeShift_{'.'.join(map(str, duration))}" for duration in self.durations
        ]

        # Add additional tokens (just Program for Structured)
        self._add_additional_tokens_to_vocab_list(vocab)

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = {
            "Pitch": {
                "Velocity"
                if self.config.use_velocities
                else "Duration"
                if self.config.using_note_duration_tokens
                else "TimeShift"
            },
            "PitchDrum": {
                "Velocity"
                if self.config.use_velocities
                else "Duration"
                if self.config.using_note_duration_tokens
                else "TimeShift"
            },
            "TimeShift": {"Pitch", "PitchDrum"},
        }
        if self.config.use_velocities:
            dic["Velocity"] = {
                "Duration" if self.config.using_note_duration_tokens else "TimeShift"
            }
        if self.config.using_note_duration_tokens:
            dic["Duration"] = {"TimeShift"}
        if self.config.use_programs:
            dic["Program"] = {"Pitch", "PitchDrum"}
            dic["TimeShift"] = {"Program"}
        return dic
