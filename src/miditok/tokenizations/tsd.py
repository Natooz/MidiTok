"""TSD (TimeShift Duration) tokenizer."""

from __future__ import annotations

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS, TIME_SIGNATURE
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_beat


class TSD(MusicTokenizer):
    r"""
    TSD (Time Shift Duration) tokenizer.

    It is similar to :ref:`MIDI-Like` but uses explicit *Duration* tokens to
    represent note durations, which have showed `better results than with *NoteOff*
    tokens <https://arxiv.org/abs/2002.00212>`_. If you specify ``use_programs`` as
    ``True`` in the config file, the tokenizer will add *Program* tokens before each
    *Pitch* tokens to specify its instrument, and will treat all tracks as a single
    stream of tokens.

    **Note:** as ``TSD`` uses *TimeShifts* events to move the time from note to note,
    it can be unsuited for tracks with pauses longer than the maximum *TimeShift*
    value. In such cases, the maximum *TimeShift* value will be used.
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when
    ``config.use_programs`` is False, only the tempos and time signatures of the first
    sequence will be decoded for the whole music.
    """

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
        previous_tick = 0
        previous_note_end = 0
        ticks_per_beat = compute_ticks_per_beat(TIME_SIGNATURE[1], time_division)
        for event in events:
            # No time shift
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

                # Time shift
                # no else here as previous might have changed with rests
                if event.time != previous_tick:
                    time_shift = event.time - previous_tick
                    for dur_value, dur_ticks in zip(
                        *self._time_ticks_to_tokens(time_shift, ticks_per_beat)
                    ):
                        all_events.append(
                            Event(
                                type_="TimeShift",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{time_shift} ticks",
                            )
                        )
                        previous_tick += dur_ticks
                previous_tick = event.time

            # Time Signature: Update ticks per beat
            if event.type_ == "TimeSig":
                ticks_per_beat = compute_ticks_per_beat(
                    int(event.value.split("/")[1]), time_division
                )

            all_events.append(event)

            # Update max offset time of the notes encountered
            if event.type_ in {
                "Pitch",
                "PitchDrum",
                "PitchIntervalTime",
                "PitchIntervalChord",
            }:
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type_ in {
                "Program",
                "Tempo",
                "TimeSig",
                "Pedal",
                "PedalOff",
                "PitchBend",
                "Chord",
            }:
                previous_note_end = max(previous_note_end, event.time)

        return all_events

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

        current_track = None  # used only when one_token_stream is False
        for si, seq in enumerate(tokens):
            # Set tracking variables
            current_tick = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
            previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
            active_pedals = {}
            ticks_per_beat = score.ticks_per_quarter

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
                if tok_type == "TimeShift":
                    current_tick += self._tpb_tokens_to_ticks[ticks_per_beat][tok_val]
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                elif tok_type in [
                    "Pitch",
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                ]:
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
                            new_note = Note(current_tick, dur, pitch, int(vel))
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
                elif tok_type == "Tempo" and si == 0:
                    tempo_changes.append(Tempo(current_tick, float(tok_val)))
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if si == 0:
                        time_signature_changes.append(
                            TimeSignature(current_tick, num, den)
                        )
                    ticks_per_beat = self._tpb_per_ts[den]
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
                            tracks[pedal_prog].pedals.append(
                                Pedal(
                                    active_pedals[pedal_prog],
                                    current_tick - active_pedals[pedal_prog],
                                )
                            )
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

                if tok_type in [
                    "Program",
                    "Tempo",
                    "TimeSig",
                    "Pedal",
                    "PedalOff",
                    "PitchBend",
                    "Chord",
                ]:
                    previous_note_end = max(previous_note_end, current_tick)

            # Add current_inst to the score and handle notes still active
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

        # NoteOn/NoteOff/Velocity
        self._add_note_tokens_to_vocab_list(vocab)

        # TimeShift
        vocab += [
            f"TimeShift_{'.'.join(map(str, self.durations[i]))}"
            for i in range(len(self.durations))
        ]

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
                else {first_note_token_type, "TimeShift"}
            )
        elif self.config.using_note_duration_tokens:
            dic["Pitch"] = {"Duration"}
        else:
            dic["Pitch"] = {first_note_token_type, "TimeShift"}
        if self.config.using_note_duration_tokens:
            dic["Duration"] = {first_note_token_type, "TimeShift"}
        dic["TimeShift"] = {first_note_token_type, "TimeShift"}
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
                        "TimeShift",
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
                    dic["TimeShift"].add(token_type)
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
            dic["TimeShift"] |= {"Chord"}
            if self.config.use_programs:
                dic["Program"].add("Chord")
            if self.config.use_pitch_intervals:
                dic["Chord"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_tempos:
            dic["TimeShift"] |= {"Tempo"}
            dic["Tempo"] = {first_note_token_type, "TimeShift"}
            if self.config.use_chords:
                dic["Tempo"] |= {"Chord"}
            if self.config.use_rests:
                dic["Tempo"].add("Rest")  # only for first token
            if self.config.use_pitch_intervals:
                dic["Tempo"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_time_signatures:
            dic["TimeShift"] |= {"TimeSig"}
            dic["TimeSig"] = {first_note_token_type, "TimeShift"}
            if self.config.use_chords:
                dic["TimeSig"] |= {"Chord"}
            if self.config.use_rests:
                dic["TimeSig"].add("Rest")  # only for first token
            if self.config.use_tempos:
                dic["TimeSig"].add("Tempo")
            if self.config.use_pitch_intervals:
                dic["TimeSig"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_sustain_pedals:
            dic["TimeShift"].add("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = {"Duration"}
                if self.config.using_note_duration_tokens:
                    dic["Duration"].add("Pedal")
                elif self.config.use_velocities:
                    dic["Duration"] = {first_note_token_type, "TimeShift"}
                    dic["Velocity"].add("Pedal")
                else:
                    dic["Duration"] = {first_note_token_type, "TimeShift"}
                    dic["Pitch"].add("Pedal")
            else:
                dic["PedalOff"] = {
                    "Pedal",
                    "PedalOff",
                    first_note_token_type,
                    "TimeShift",
                }
                dic["Pedal"] = {"Pedal", first_note_token_type, "TimeShift"}
                dic["TimeShift"].add("PedalOff")
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
            dic["PitchBend"] = {first_note_token_type, "TimeShift"}
            if self.config.use_programs and not self.config.program_changes:
                dic["Program"].add("PitchBend")
            else:
                dic["TimeShift"].add("PitchBend")
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
            dic["Rest"] = {"Rest", first_note_token_type, "TimeShift"}
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
        else:
            dic["TimeShift"].add("TimeShift")

        if self.config.program_changes:
            for token_type in {
                "TimeShift",
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
