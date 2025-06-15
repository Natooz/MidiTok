"""MuMIDI tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from symusic import Note, Score, Tempo, Track

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import detect_chords, get_bars_ticks, get_score_ticks_per_beat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np


class MuMIDI(MusicTokenizer):
    r"""
    MuMIDI tokenizer.

    Introduced with `PopMAG (Ren et al.) <https://arxiv.org/abs/2008.07703>`_,
    this tokenization made for multitrack tasks and uses embedding pooling. Time is
    represented with *Bar* and *Position* tokens. The key idea of MuMIDI is to represent
    all tracks in a single token sequence. At each time step, *Track* tokens preceding
    note tokens indicate their track. MuMIDI also include a "built-in" and learned
    positional encoding. As in the original paper, the pitches of drums are distinct
    from those of all other instruments.
    Each pooled token will be a list of the form (index: Token type):

    * 0: Pitch / PitchDrum / Position / Bar / Program / (Chord) / (Rest);
    * 1: BarPosEnc;
    * 2: PositionPosEnc;
    * (-3 / 3: Tempo);
    * -2: Velocity;
    * -1: Duration.

    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can
    be very delicate. Hence, we do not recommend this tokenization for generation with
    small models.

    **Notes:**

    * Tokens are first sorted by time, then track, then pitch values.
    * Tracks with the same *Program* will be merged.
    """

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_rests = False
        self.config.use_time_signatures = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_programs = True
        self.config.use_pitch_intervals = True
        self.config.one_token_stream_for_programs = True
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
                "MuMIDI only allows to use note duration tokens for either all "
                "programs or none.",
                stacklevel=2,
            )

        if "max_bar_embedding" not in self.config.additional_params:
            self.config.additional_params["max_bar_embedding"] = 60

        self.vocab_types_idx = {
            "Pitch": 0,
            "PitchDrum": 0,
            "Position": 0,
            "Bar": 0,
            "Program": 0,
            "BarPosEnc": 1,
            "PositionPosEnc": 2,
        }
        if self.config.use_velocities:
            self.vocab_types_idx["Velocity"] = -1
        if self.config.using_note_duration_tokens:
            self.vocab_types_idx["Duration"] = -1
            if self.config.use_velocities:
                self.vocab_types_idx["Velocity"] = -2
        if self.config.use_chords:
            self.vocab_types_idx["Chord"] = 0
        if self.config.use_rests:
            self.vocab_types_idx["Rest"] = 0
        if self.config.use_tempos:
            self.vocab_types_idx["Tempo"] = -3

    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        """
        Create the time events from a list of global and track events.

        Unused here.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        MuMIDI has its own implementation and doesn't use ``_add_time_events``.

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

        # Convert each track to tokens (except first pos to track time)
        if self.config.use_chords:
            ticks_per_beat = get_score_ticks_per_beat(score)
        else:
            ticks_per_beat = None
        note_tokens = []
        for track in score.tracks:
            if track.program in self.config.programs:
                note_tokens += self._track_to_tokens(track, ticks_per_beat)

        note_tokens.sort(
            key=lambda x: (x[0].time, x[0].desc)
        )  # Sort by time then track

        ticks_per_sample = score.ticks_per_quarter / self.config.max_num_pos_per_beat
        ticks_per_bar = score.ticks_per_quarter * 4
        tokens = []

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track = -2  # because -2 doesn't exist
        current_tempo_idx = 0
        current_tempo = round(score.tempos[current_tempo_idx].tempo, 2)
        for note_token in note_tokens:
            # (Tempo) update tempo values current_tempo
            # If the current tempo is not the last one
            if self.config.use_tempos and current_tempo_idx + 1 < len(score.tempos):
                # Will loop over incoming tempo changes
                for tempo_change in score.tempos[current_tempo_idx + 1 :]:
                    # If this tempo change happened before the current moment
                    if tempo_change.time <= note_token[0].time:
                        current_tempo = round(tempo_change.tempo, 2)
                        current_tempo_idx += (
                            1  # update tempo value (might not change) and index
                        )
                    elif tempo_change.time > note_token[0].time:
                        break  # this tempo change is beyond the current time step
            # Positions and bars pos enc
            if note_token[0].time != current_tick:
                pos_index = int((note_token[0].time % ticks_per_bar) / ticks_per_sample)
                current_tick = note_token[0].time
                current_pos = pos_index
                current_track = -2  # reset
                # (New bar)
                if current_bar < current_tick // ticks_per_bar:
                    num_new_bars = current_tick // ticks_per_bar - current_bar
                    for i in range(num_new_bars):
                        bar_token = [
                            "Bar_None",
                            f"BarPosEnc_{current_bar + i + 1}",
                            "PositionPosEnc_None",
                        ]
                        if self.config.use_tempos:
                            bar_token.append(f"Tempo_{current_tempo}")
                        tokens.append(bar_token)
                    current_bar += num_new_bars
                # Position
                pos_token = [
                    f"Position_{current_pos}",
                    f"BarPosEnc_{current_bar}",
                    f"PositionPosEnc_{current_pos}",
                ]
                if self.config.use_tempos:
                    pos_token.append(f"Tempo_{current_tempo}")
                tokens.append(pos_token)
            # Program (track)
            if note_token[0].desc != current_track:
                current_track = note_token[0].desc
                track_token = [
                    f"Program_{current_track}",
                    f"BarPosEnc_{current_bar}",
                    f"PositionPosEnc_{current_pos}",
                ]
                if self.config.use_tempos:
                    track_token.append(f"Tempo_{current_tempo}")
                tokens.append(track_token)

            # Adding bar and position tokens to notes for positional encoding
            note_token[0] = str(note_token[0])
            note_token.insert(1, f"BarPosEnc_{current_bar}")
            note_token.insert(2, f"PositionPosEnc_{current_pos}")
            if self.config.use_tempos:
                note_token.insert(3, f"Tempo_{current_tempo}")
            tokens.append(note_token)

        tokens = TokSequence(tokens=tokens)
        self.complete_sequence(tokens)
        return tokens

    def _track_to_tokens(
        self, track: Track, ticks_per_beat: np.ndarray = None
    ) -> list[list[Event]]:
        r"""
        Convert a track (``symusic.Track``) into a sequence of tokens.

        For each note, it creates a time step as a
        list of tokens where (list index: token type):
        * 0: Pitch (as an Event object for sorting purpose afterward);
        * 1: Velocity;
        * 2: Duration.

        :param track: track object to convert.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            time signature denominator section. The numbers of ticks per beat depend on
            the time signatures of the ``Score`` being parsed. The array has a shape
            ``(N,2)``, for ``N`` changes of ticks per beat, and the second dimension
            representing the end tick of each section and the number of ticks per beat
            respectively. Only used when using chords. (default: ``None``)
        :return: sequence of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by
        # pitch: notes.sort(key=lambda x: (x.start, x.pitch)) (done in preprocess_score)

        tokens = []
        tpb = self.time_division
        for note in track.notes:
            # Note
            duration = note.end - note.start
            new_token = [
                Event(
                    type_="Pitch" if not track.is_drum else "PitchDrum",
                    value=note.pitch,
                    time=note.start,
                    desc=track.program if not track.is_drum else -1,
                ),
            ]
            if self.config.use_velocities:
                new_token.append(f"Velocity_{note.velocity}")
            if self.config.using_note_duration_tokens:
                dur_token = self._tpb_ticks_to_tokens[tpb][duration]
                new_token.append(f"Duration_{dur_token}")
            tokens.append(new_token)

        # Adds chord tokens if specified
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                ticks_per_beat,
                chord_maps=self.config.chord_maps,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_num_notes_range=self.config.chord_unknown,
            )
            unsqueezed = []
            for c in range(len(chords)):
                chords[c].desc = track.program
                unsqueezed.append([chords[c]])
            tokens = (
                unsqueezed + tokens
            )  # chords at the beginning to keep the good order during sorting

        return tokens

    def _tokens_to_score(
        self,
        tokens: TokSequence,
        _: None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param _: in place of programs of the parent method, unused here.
            (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        score = Score(self.time_division)

        # Tempos
        if self.config.use_tempos and len(tokens) > 0:
            first_tempo = float(tokens.tokens[0][3].split("_")[1])
        else:
            first_tempo = self.default_tempo
        score.tempos.append(Tempo(0, first_tempo))

        tracks = {}
        current_tick = 0
        current_bar = -1
        current_track = 0  # default set to piano
        ticks_per_beat = score.ticks_per_quarter
        for time_step in tokens.tokens:
            tok_type, tok_val = time_step[0].split("_")
            if tok_type == "Bar":
                current_bar += 1
                current_tick = current_bar * ticks_per_beat * 4
            elif tok_type == "Position":
                if current_bar == -1:
                    current_bar = (
                        0  # as this Position token occurs before any Bar token
                    )
                current_tick = current_bar * ticks_per_beat * 4 + int(tok_val)
            elif tok_type == "Program":
                current_track = tok_val
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif tok_type in {"Pitch", "PitchDrum"}:
                vel = (
                    time_step[self.vocab_types_idx["Velocity"]].split("_")[1]
                    if self.config.use_velocities
                    else DEFAULT_VELOCITY
                )
                duration = (
                    time_step[-1].split("_")[1]
                    if self.config.using_note_duration_tokens
                    else int(self.config.default_note_duration * ticks_per_beat)
                )
                if any(val == "None" for val in (vel, duration)):
                    continue
                pitch = int(tok_val)
                vel = int(vel)
                if isinstance(duration, str):
                    duration = self._tpb_tokens_to_ticks[ticks_per_beat][duration]

                tracks[current_track].append(Note(current_tick, duration, pitch, vel))

            # Decode tempo if required
            if self.config.use_tempos:
                tempo_val = float(time_step[3].split("_")[1])
                if tempo_val != score.tempos[-1].tempo:
                    score.tempos.append(Tempo(current_tick, tempo_val))

        # Appends created notes to Score object
        for program, notes in tracks.items():
            if int(program) == -1:
                score.tracks.append(Track(name="Drums", program=0, is_drum=True))
            else:
                score.tracks.append(
                    Track(
                        name=MIDI_INSTRUMENTS[int(program)]["name"],
                        program=int(program),
                        is_drum=False,
                    )
                )
            score.tracks[-1].notes = notes

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

        For MUMIDI, token index 0 is used as a padding index for training.
        * 0: Pitch / PitchDrum / Position / Bar / Program / (Chord) / (Rest)
        * 1: BarPosEnc
        * 2: PositionPosEnc
        * (-3 / 3: Tempo)
        * -2: Velocity
        * -1: Duration

        :return: the vocabulary as a list of string.
        """
        vocab = [[] for _ in range(3)]

        # PITCH & DRUM PITCHES & BAR & POSITIONS & PROGRAM
        vocab[0] += [
            f"Pitch_{i}"
            for i in range(self.config.pitch_range[0], self.config.pitch_range[1] + 1)
        ]
        vocab[0] += [
            f"PitchDrum_{i}"
            for i in range(
                self.config.drums_pitch_range[0],
                self.config.drums_pitch_range[1] + 1,
            )
        ]
        vocab[0] += ["Bar_None"]  # new bar token
        max_num_beats = max(ts[0] for ts in self.time_signatures)
        num_positions = self.config.max_num_pos_per_beat * max_num_beats
        vocab[0] += [f"Position_{i}" for i in range(num_positions)]
        vocab[0] += [f"Program_{program}" for program in self.config.programs]

        # BAR POS ENC
        vocab[1] += [
            f"BarPosEnc_{i}"
            for i in range(self.config.additional_params["max_bar_embedding"])
        ]

        # POSITION POS ENC
        vocab[2] += [
            "PositionPosEnc_None"
        ]  # special embedding used with 'Bar_None' tokens
        vocab[2] += [f"PositionPosEnc_{i}" for i in range(num_positions)]  # pos enc

        # CHORD
        if self.config.use_chords:
            vocab[0] += self._create_chords_tokens()

        # REST
        if self.config.use_rests:
            vocab[0] += [f"Rest_{'.'.join(map(str, rest))}" for rest in self.rests]

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # Velocity and Duration in last position
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

        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = {
            "Bar": {"Bar", "Position"},
            "Position": {"Program"},
            "Program": {"Pitch", "PitchDrum"},
            "Pitch": {"Pitch", "Program", "Bar", "Position"},
            "PitchDrum": {"PitchDrum", "Program", "Bar", "Position"},
        }

        if self.config.use_chords:
            dic["Program"] |= {"Chord"}
            dic["Chord"] = {"Pitch"}

        return dic

    def _tokens_errors(self, tokens: list[list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        This method is intended to be overridden by tokenizer classes. The
        implementation in the ``MusicTokenizer`` class will check token types,
        duplicated notes and time errors. It works for ``REMI``, ``TSD`` and
        ``Structured``.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err = 0
        previous_type = tokens[0][0].split("_")[0]
        current_pitches = []
        current_bar = int(tokens[0][1].split("_")[1])
        current_pos = tokens[0][2].split("_")[1]
        current_pos = int(current_pos) if current_pos != "None" else -1

        for token in tokens[1:]:
            bar_value = int(token[1].split("_")[1])
            pos_value = token[2].split("_")[1]
            pos_value = int(pos_value) if pos_value != "None" else -1
            token_type, token_value = token[0].split("_")

            if any(tok.split("_")[0] in ["PAD", "MASK"] for i, tok in enumerate(token)):
                err += 1
                continue

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Bar":
                    current_bar += 1
                    current_pos = -1
                    current_pitches = []
                elif self.config.remove_duplicated_notes and token_type == "Pitch":
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == "Position":
                    if int(token_value) <= current_pos or int(token_value) != pos_value:
                        err += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = []
                elif token_type == "Program":
                    current_pitches = []

                if pos_value < current_pos or bar_value < current_bar:
                    err += 1
            # Bad token type
            else:
                err += 1

            previous_type = token_type
        return err
