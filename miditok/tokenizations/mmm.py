from __future__ import annotations

from copy import deepcopy
from typing import Any, List, cast

import numpy as np
from symusic import Note, Score, Tempo, TimeSignature, Track

from ..classes import Event, TokSequence
from ..constants import (
    MIDI_INSTRUMENTS,
    MMM_DENSITY_BINS_MAX,
    TIME_SIGNATURE,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq


class MMM(MIDITokenizer):
    r"""MMM, standing for `Multi-Track Music Machine <https://arxiv.org/abs/2008.06048>`_,
    is a multitrack tokenization primarily designed for music inpainting and infilling.
    Tracks are tokenized independently and concatenated into a single token sequence.
    ``Bar_Fill`` tokens are used to specify the bars to fill (or inpaint, or rewrite),
    the new tokens are then autoregressively generated.
    Note that *this implementation represents note durations with ``Duration`` tokens*
    instead of the ``NoteOff`` strategy of the `original paper <https://arxiv.org/abs/2008.06048>`_.
    The reason being that ``NoteOff`` tokens perform poorer for generation with causal
    models.

    **Add a `density_bins_max` entry in the config, mapping to a tuple specifying the
    number of density bins, and the maximum density in notes per beat to consider.
    (default: (10, 20))**

    **Note:** When decoding tokens with tempos, only the tempos of the first track
    will be decoded.
    """

    def _tweak_config_before_creating_voc(self):
        self.config.one_token_stream_for_programs = True
        self.config.program_changes = False
        self.config.use_programs = True
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        # Recreate densities here just in case density_bins_max was loaded from params
        # (list to np array)
        if "density_bins_max" not in self.config.additional_params:
            self.config.additional_params["density_bins_max"] = MMM_DENSITY_BINS_MAX
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

    def _add_time_events(self, events: list[Event]) -> list[Event]:
        r"""Internal method intended to be implemented by inheriting classes.
        It creates the time events from the list of global and track events, and as
        such the final token sequence.

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        # Creates first events
        all_events = [Event("Bar", "Start", 0)]

        # Time events
        time_sig_change = TimeSignature(0, *TIME_SIGNATURE)
        ticks_per_bar = self._compute_ticks_per_bar(
            time_sig_change, self._time_division
        )
        bar_at_last_ts_change = 0
        previous_tick = 0
        current_bar = 0
        tick_at_last_ts_change = 0
        for ei in range(len(events)):
            if events[ei].type == "TimeSig":
                bar_at_last_ts_change += (
                    events[ei].time - tick_at_last_ts_change
                ) // ticks_per_bar
                tick_at_last_ts_change = events[ei].time
                ticks_per_bar = self._compute_ticks_per_bar(
                    TimeSignature(
                        events[ei].time, *list(map(int, events[ei].value.split("/")))
                    ),
                    self._time_division,
                )
            if events[ei].time != previous_tick:
                # Bar
                nb_new_bars = (
                    bar_at_last_ts_change
                    + (events[ei].time - tick_at_last_ts_change) // ticks_per_bar
                    - current_bar
                )
                if nb_new_bars > 0:
                    for i in range(nb_new_bars):
                        all_events += [
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
                    tick_at_current_bar = (
                        tick_at_last_ts_change
                        + (current_bar - bar_at_last_ts_change) * ticks_per_bar
                    )
                    previous_tick = tick_at_current_bar

                # TimeShift
                if events[ei].time != previous_tick:
                    time_shift = events[ei].time - previous_tick
                    index = np.argmin(np.abs(self._durations_ticks - time_shift))
                    all_events.append(
                        Event(
                            type="TimeShift",
                            value=".".join(map(str, self.durations[index])),
                            time=previous_tick,
                            desc=f"{time_shift} ticks",
                        )
                    )

                previous_tick = events[ei].time

            # Add the event to the new list
            all_events.append(events[ei])

        all_events += [
            Event("Bar", "End", all_events[-1].time + 1),
            Event("Track", "End", all_events[-1].time + 1),
        ]
        return all_events

    def _midi_to_tokens(self, midi: Score) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        Tokenization treating all tracks as a single token sequence might
        override this method, e.g. Octuple or PopMAG.

        :param midi: the MIDI object to convert.
        :return: sequences of tokens.
        """
        note_density_bins = self.config.additional_params["note_densities"]

        # Create events list
        all_events = []

        # Global events (Tempo, TimeSignature)
        global_events = self._create_midi_events(midi)

        # Adds track tokens
        # Disable use_programs so that _create_track_events do not add Program events
        self.config.use_programs = False
        for track in midi.tracks:
            note_density = len(track.notes) / (
                max(note.end for note in track.notes) / midi.ticks_per_quarter
            )
            note_density_idx = np.argmin(np.abs(note_density_bins - note_density))
            note_density = int(note_density_bins[note_density_idx])
            all_events += [
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
            ]

            track_events = deepcopy(global_events) + self._create_track_events(track)
            track_events.sort(key=lambda x: x.time)
            all_events += self._add_time_events(track_events)
            all_events.append(Event("Track", "End", all_events[-1].time + 1))

        self.config.use_programs = True
        tok_sequence = TokSequence(events=all_events)
        self.complete_sequence(tok_sequence)
        return tok_sequence

    def _tokens_to_midi(
        self,
        tokens: TokSequence | list | np.ndarray | Any,
        _=None,
        time_division: int | None = None,
    ) -> Score:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`,
        :param _: unused, to match parent method signature
        :param time_division: MIDI time division / resolution, in ticks/beat (of the
            MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        if time_division is None:
            time_division = self._time_division
        tokens = cast(TokSequence, tokens)
        midi = Score(time_division)
        if time_division % max(self.config.beat_res.values()) != 0:
            raise ValueError(
                f"Invalid time division, please give one divisible by"
                f"{max(self.config.beat_res.values())}"
            )
        tokens = cast(List[str], tokens.tokens)  # for reducing type errors

        # RESULTS
        tracks: list[Track] = []
        tempo_changes = []
        time_signature_changes = []
        ticks_per_bar = self._compute_ticks_per_bar(
            TimeSignature(0, *TIME_SIGNATURE), time_division
        )

        current_tick = tick_at_current_bar = 0
        current_bar = -1
        previous_note_end = 0  # unused (rest)
        first_program = None
        current_program = -2
        previous_pitch_onset = previous_pitch_chord = -128
        for ti, token in enumerate(tokens):
            tok_type, tok_val = token.split("_")
            if tok_type == "Program":
                current_program = int(tok_val)
                tracks.append(
                    Track(
                        program=0 if current_program == -1 else current_program,
                        is_drum=current_program == -1,
                        name="Drums"
                        if current_program == -1
                        else MIDI_INSTRUMENTS[current_program]["name"],
                    )
                )
                if first_program is None:
                    first_program = current_program
                current_tick = 0
                current_bar = -1
                previous_note_end = 0
            elif token == "Bar_Start":  # noqa: S105
                current_bar += 1
                if current_bar > 0:
                    current_tick = tick_at_current_bar + ticks_per_bar
                tick_at_current_bar = current_tick
            elif tok_type == "TimeShift":
                if current_bar == -1:
                    # as this Position token occurs before any Bar token
                    current_bar = 0
                current_tick += self._token_duration_to_ticks(tok_val, time_division)
            elif tok_type == "Tempo" and (
                first_program is None or current_program == first_program
            ):
                tempo_changes.append(Tempo(current_tick, float(token.split("_")[1])))
            elif tok_type == "TimeSig" and (
                first_program is None or current_program == first_program
            ):
                num, den = self._parse_token_time_signature(token.split("_")[1])
                time_signature_changes.append(TimeSignature(current_tick, num, den))
                ticks_per_bar = self._compute_ticks_per_bar(
                    time_signature_changes[-1], time_division
                )
            elif tok_type in ["Pitch", "PitchIntervalTime", "PitchIntervalChord"]:
                if tok_type == "Pitch":
                    pitch = int(tok_val)
                    previous_pitch_onset = pitch
                    previous_pitch_chord = pitch
                # We update previous_pitch_onset and previous_pitch_chord even if the
                # try fails.
                elif tok_type == "PitchIntervalTime":
                    pitch = previous_pitch_onset + int(tok_val)
                    previous_pitch_onset = pitch
                    previous_pitch_chord = pitch
                else:  # PitchIntervalChord
                    pitch = previous_pitch_chord + int(tok_val)
                    previous_pitch_chord = pitch
                try:
                    vel_type, vel = tokens[ti + 1].split("_")
                    dur_type, dur = tokens[ti + 2].split("_")
                    if vel_type == "Velocity" and dur_type == "Duration":
                        dur = self._token_duration_to_ticks(dur, time_division)
                        tracks[-1].notes.append(
                            Note(current_tick, dur, pitch, int(vel))
                        )
                        previous_note_end = max(previous_note_end, current_tick + dur)
                except IndexError:
                    # A well constituted sequence should not raise an exception
                    # However with generated sequences this can happen, or if the
                    # sequence isn't finished
                    pass

        # create MidiFile
        midi.tracks = tracks
        midi.tempos = tempo_changes
        midi.time_signatures = time_signature_changes

        return midi

    def _create_base_vocabulary(self) -> list[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an
        underscore. Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MIDITokenizer`.

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

        # Add additional tokens (handles Programs too)
        self._add_additional_tokens_to_vocab_list(vocab)

        return vocab

    def _create_token_types_graph(self) -> dict[str, list[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.

        :return: the token types transitions dictionary
        """
        dic: dict[str, list[str]] = {
            "Bar": ["Bar", "TimeShift", "Pitch", "Track"],
            "TimeShift": ["Pitch"],
            "Track": ["Program", "Track"],
            "Program": ["NoteDensity"],
            "NoteDensity": ["Bar"],
            "Pitch": ["Velocity"],
            "Velocity": ["Duration"],
            "Duration": ["Pitch", "TimeShift", "Bar"],
        }

        if self.config.use_pitch_intervals:
            for token_type in ("PitchIntervalTime", "PitchIntervalChord"):
                dic[token_type] = ["Velocity"]
                dic["Duration"].append(token_type)
                dic["TimeShift"].append(token_type)
                dic["Bar"].append(token_type)

        if self.config.use_time_signatures:
            dic["Bar"] += ["TimeSig"]
            dic["TimeSig"] = ["Pitch", "TimeShift", "Bar"]
            if self.config.use_pitch_intervals:
                dic["TimeSig"] += ["PitchIntervalTime", "PitchIntervalChord"]

        if self.config.use_chords:
            dic["Chord"] = ["TimeShift", "Pitch"]
            dic["Bar"] += ["Chord"]
            dic["TimeShift"] += ["Chord"]
            if self.config.use_pitch_intervals:
                dic["Chord"] += ["PitchIntervalTime", "PitchIntervalChord"]

        if self.config.use_tempos:
            dic["Tempo"] = ["TimeShift", "Pitch", "Bar"]
            dic["Bar"] += ["Tempo"]
            dic["TimeShift"] += ["Tempo"]
            if self.config.use_time_signatures:
                dic["TimeSig"] += ["Tempo"]
            if self.config.use_chords:
                dic["Tempo"] += ["Chord"]
            if self.config.use_pitch_intervals:
                dic["Tempo"] += ["PitchIntervalTime", "PitchIntervalChord"]

        dic["Fill"] = list(dic.keys())

        return dic

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(
        self, tokens_to_check: TokSequence | list[int | list[int]]
    ) -> float:
        tokens_to_check = cast(TokSequence, tokens_to_check)
        nb_tok_predicted = len(tokens_to_check)  # used to norm the score
        if nb_tok_predicted == 0:
            return 0
        if self.has_bpe:
            self.decode_bpe(tokens_to_check)
        self.complete_sequence(tokens_to_check)

        # Override from here
        tokens = cast(List[str], tokens_to_check.tokens)
        note_tokens_types = ["Pitch"]
        if self.config.use_pitch_intervals:
            note_tokens_types += ["PitchIntervalTime", "PitchIntervalChord"]

        err_type = 0  # i.e. incompatible next type predicted
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_pitches = []
        previous_pitch_onset = previous_pitch_chord = -128

        # Init first note and current pitches if needed
        if previous_type == "Pitch":
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches.append(pitch_val)

        for token in tokens[1:]:
            # err_tokens = tokens[i - 4 : i + 4]  # uncomment for debug
            event_type, event_value = token.split("_")[0], token.split("_")[1]

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type in ["Bar", "TimeShift"]:  # reset
                    current_pitches = []
                elif (
                    self.config.remove_duplicated_notes
                    and event_type in note_tokens_types
                ):
                    if event_type == "Pitch":
                        pitch_val = int(event_value)
                        previous_pitch_onset = pitch_val
                        previous_pitch_chord = pitch_val
                    elif event_type == "PitchIntervalTime":
                        pitch_val = previous_pitch_onset + int(event_value)
                        previous_pitch_onset = pitch_val
                        previous_pitch_chord = pitch_val
                    else:  # PitchIntervalChord
                        pitch_val = previous_pitch_chord + int(event_value)
                        previous_pitch_chord = pitch_val
                    if pitch_val in current_pitches:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches.append(pitch_val)
                elif event_type == "Program":  # reset
                    current_pitches = []
                    previous_pitch_onset = previous_pitch_chord = -128
            # Bad token type
            else:
                err_type += 1
            previous_type = event_type

        return (err_type + err_note) / nb_tok_predicted
