from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange

from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..classes import TokSequence, Event, TokenizerConfig
from ..utils import detect_chords
from ..constants import (
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    DRUM_PITCH_RANGE,
)


class MuMIDI(MIDITokenizer):
    r"""Introduced with `PopMAG (Ren et al.) <https://arxiv.org/abs/2008.07703>`_,
    this tokenization made for multitrack tasks and uses embedding pooling. Time is
    represented with *Bar* and *Position* tokens. The key idea of MuMIDI is to represent
    all tracks in a single token sequence. At each time step, *Track* tokens preceding
    note tokens indicate their track. MuMIDI also include a "built-in" and learned
    positional encoding. As in the original paper, the pitches of drums are distinct
    from those of all other instruments.
    Each pooled token will be a list of the form (index: Token type):
    * 0: Pitch / DrumPitch / Position / Bar / Program / (Chord) / (Rest)
    * 1: BarPosEnc
    * 2: PositionPosEnc
    * (-3 / 3: Tempo)
    * -2: Velocity
    * -1: Duration

    The output hidden states of the model will then be fed to several output layers
    (one per token type). This means that the training requires to add multiple losses.
    For generation, the decoding implies sample from several distributions, which can be
    very delicate. Hence, we do not recommend this tokenization for generation with small models.

    **Notes:**
        * Tokens are first sorted by time, then track, then pitch values.
        * Tracks with the same *Program* will be merged.

    :param tokenizer_config: the tokenizer's configuration, as a :class:`miditok.classes.TokenizerConfig` object.
    :param drum_pitch_range: range of used MIDI pitches for drums exclusively
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        drum_pitch_range: Tuple[int, int] = DRUM_PITCH_RANGE,
        params: Union[str, Path] = None,
    ):
        if tokenizer_config is not None:
            if "drum_pitch_range" not in tokenizer_config.additional_params:
                tokenizer_config.additional_params[
                    "drum_pitch_range"
                ] = drum_pitch_range
            if "max_bar_embedding" not in tokenizer_config.additional_params:
                # this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
                tokenizer_config.additional_params["max_bar_embedding"] = 60
        super().__init__(tokenizer_config, True, params=params)

    def _tweak_config_before_creating_voc(self):
        self.config.use_rests = False
        self.config.use_time_signatures = False
        # self.one_token_stream = True

        self.vocab_types_idx = {
            "Pitch": 0,
            "DrumPitch": 0,
            "Position": 0,
            "Bar": 0,
            "Program": 0,
            "BarPosEnc": 1,
            "PositionPosEnc": 2,
            "Velocity": -2,
            "Duration": -1,
        }
        if self.config.use_chords:
            self.vocab_types_idx["Chord"] = 0
        if self.config.use_rests:
            self.vocab_types_idx["Rest"] = 0
        if self.config.use_tempos:
            self.vocab_types_idx["Tempo"] = -3

    @_out_as_complete_seq
    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Tokenize a MIDI file.
        Each pooled token will be a list of the form (index: Token type):
        * 0: Pitch / DrumPitch / Position / Bar / Program / (Chord) / (Rest)
        * 1: BarPosEnc
        * 2: PositionPosEnc
        * (-3 / 3: Tempo)
        * -2: Velocity
        * -1: Duration

        :param midi: the MIDI object to convert
        :return: sequences of tokens
        """
        # Check bar embedding limit, update if needed
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.config.additional_params["max_bar_embedding"] < nb_bars:
            for i in range(self.config.additional_params["max_bar_embedding"], nb_bars):
                self.add_to_vocab(f"BarPosEnc_{i}", 1)
            self.config.additional_params["max_bar_embedding"] = nb_bars

        # Convert each track to tokens (except first pos to track time)
        note_tokens = []
        for track in midi.instruments:
            if track.program in self.config.programs:
                note_tokens += self.track_to_tokens(track)

        note_tokens.sort(
            key=lambda x: (x[0].time, x[0].desc)
        )  # Sort by time then track

        ticks_per_sample = midi.ticks_per_beat / max(self.config.beat_res.values())
        ticks_per_bar = midi.ticks_per_beat * 4
        tokens = []

        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_track = -2  # because -2 doesn't exist
        current_tempo_idx = 0
        current_tempo = self._current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        for note_token in note_tokens:
            # (Tempo) update tempo values current_tempo
            if self.config.use_tempos:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(
                    self._current_midi_metadata["tempo_changes"]
                ):
                    # Will loop over incoming tempo changes
                    for tempo_change in self._current_midi_metadata["tempo_changes"][
                        current_tempo_idx + 1 :
                    ]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note_token[0].time:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += (
                                1  # update tempo value (might not change) and index
                            )
                        elif tempo_change.time > note_token[0].time:
                            break  # this tempo change is beyond the current time step, we break the loop
            # Positions and bars pos enc
            if note_token[0].time != current_tick:
                pos_index = int((note_token[0].time % ticks_per_bar) / ticks_per_sample)
                current_tick = note_token[0].time
                current_pos = pos_index
                current_track = -2  # reset
                # (New bar)
                if current_bar < current_tick // ticks_per_bar:
                    nb_new_bars = current_tick // ticks_per_bar - current_bar
                    for i in range(nb_new_bars):
                        bar_token = [
                            "Bar_None",
                            f"BarPosEnc_{current_bar + i + 1}",
                            "PositionPosEnc_None",
                        ]
                        if self.config.use_tempos:
                            bar_token.append(f"Tempo_{current_tempo}")
                        tokens.append(bar_token)
                    current_bar += nb_new_bars
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

        return TokSequence(tokens=tokens)

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, str]]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).
        For each note, it creates a time step as a list of tokens where (list index: token type):
        * 0: Pitch (as an Event object for sorting purpose afterwards)
        * 1: Velocity
        * 2: Duration

        :param track: track object to convert.
        :return: sequence of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]

        tokens = []
        for note in track.notes:
            # Note
            duration = note.end - note.start
            dur_idx = np.argmin(np.abs(dur_bins - duration))
            if not track.is_drum:
                tokens.append(
                    [
                        Event(
                            type="Pitch",
                            value=note.pitch,
                            time=note.start,
                            desc=track.program,
                        ),
                        f"Velocity_{note.velocity}",
                        f'Duration_{".".join(map(str, self.durations[dur_idx]))}',
                    ]
                )
            else:
                tokens.append(
                    [
                        Event(
                            type="DrumPitch",
                            value=note.pitch,
                            time=note.start,
                            desc=-1,
                        ),
                        f"Velocity_{note.velocity}",
                        f'Duration_{".".join(map(str, self.durations[dur_idx]))}',
                    ]
                )

        # Adds chord tokens if specified
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                self._current_midi_metadata["time_division"],
                chord_maps=self.config.chord_maps,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_nb_notes_range=self.config.chord_unknown,
            )
            unsqueezed = []
            for c in range(len(chords)):
                chords[c].desc = track.program
                unsqueezed.append([chords[c]])
            tokens = (
                unsqueezed + tokens
            )  # chords at the beginning to keep the good order during sorting

        return tokens

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        _=None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        A time step is a list of tokens where (list index: token type):
        * 0: Pitch / DrumPitch / Position / Bar / Program / (Chord) / (Rest)
        * 1: BarPosEnc
        * 2: PositionPosEnc
        * (-3 / 3: Tempo)
        * -2: Velocity
        * -1: Duration

        :param tokens: tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param _: unused, to match parent method signature
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert (
            time_division % max(self.config.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.config.beat_res.values())}"
        midi = MidiFile(ticks_per_beat=time_division)

        # Tempos
        if self.config.use_tempos:
            first_tempo = int(tokens.tokens[0][3].split("_")[1])
        else:
            first_tempo = TEMPO
        midi.tempo_changes.append(TempoChange(first_tempo, 0))

        ticks_per_sample = time_division // max(self.config.beat_res.values())
        tracks = {}
        current_tick = 0
        current_bar = -1
        current_track = 0  # default set to piano
        for time_step in tokens.tokens:
            tok_type, tok_val = time_step[0].split("_")
            if tok_type == "Bar":
                current_bar += 1
                current_tick = current_bar * time_division * 4
            elif tok_type == "Position":
                if current_bar == -1:
                    current_bar = (
                        0  # as this Position token occurs before any Bar token
                    )
                current_tick = (
                    current_bar * time_division * 4 + int(tok_val) * ticks_per_sample
                )
            elif tok_type == "Program":
                current_track = tok_val
                try:
                    _ = tracks[current_track]
                except KeyError:
                    tracks[current_track] = []
            elif tok_type == "Pitch" or tok_type == "DrumPitch":
                vel, duration = (time_step[i].split("_")[1] for i in (-2, -1))
                if any(val == "None" for val in (vel, duration)):
                    continue
                pitch = int(tok_val)
                vel = int(vel)
                duration = self._token_duration_to_ticks(duration, time_division)

                tracks[current_track].append(
                    Note(vel, pitch, current_tick, current_tick + duration)
                )

            # Decode tempo if required
            if self.config.use_tempos:
                tempo_val = int(time_step[3].split("_")[1])
                if tempo_val != midi.tempo_changes[-1].tempo:
                    midi.tempo_changes.append(TempoChange(tempo_val, current_tick))

        # Appends created notes to MIDI object
        for program, notes in tracks.items():
            if int(program) == -1:
                midi.instruments.append(Instrument(0, True, "Drums"))
            else:
                midi.instruments.append(
                    Instrument(
                        int(program), False, MIDI_INSTRUMENTS[int(program)]["name"]
                    )
                )
            midi.instruments[-1].notes = notes

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def tokens_to_track(
        self,
        tokens: TokSequence,
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ):
        r"""Not relevant / implemented for MuMIDI. Use :py:meth:`miditok.MuMIDI.tokens_to_midi` instead.

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        pass

    def _create_base_vocabulary(self, sos_eos_tokens: bool = None) -> List[List[str]]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        For MUMIDI, token index 0 is used as a padding index for training.
        * 0: Pitch / DrumPitch / Position / Bar / Program / (Chord) / (Rest)
        * 1: BarPosEnc
        * 2: PositionPosEnc
        * (-3 / 3: Tempo)
        * -2: Velocity
        * -1: Duration

        :return: the vocabulary as a list of string.
        """
        vocab = [[] for _ in range(3)]

        # PITCH & DRUM PITCHES & BAR & POSITIONS & PROGRAM
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        vocab[0] += [
            f"DrumPitch_{i}"
            for i in range(*self.config.additional_params["drum_pitch_range"])
        ]
        vocab[0] += ["Bar_None"]  # new bar token
        nb_positions = max(self.config.beat_res.values()) * 4  # 4/* time signature
        vocab[0] += [f"Position_{i}" for i in range(nb_positions)]
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
        vocab[2] += [f"PositionPosEnc_{i}" for i in range(nb_positions)]  # pos enc

        # CHORD
        if self.config.use_chords:
            vocab[0] += self._create_chords_tokens()

        # REST
        if self.config.use_rests:
            vocab[0] += [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # Velocity and Duration in last position
        # VELOCITY
        vocab.append([f"Velocity_{i}" for i in self.velocities])

        # DURATION
        vocab.append(
            [f'Duration_{".".join(map(str, duration))}' for duration in self.durations]
        )

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Here the combination of Pitch, Velocity and Duration tokens is represented by
        "Pitch" in the graph.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Bar"] = ["Bar", "Position"]
        dic["Position"] = ["Program"]
        dic["Program"] = ["Pitch", "DrumPitch"]
        dic["Pitch"] = ["Pitch", "Program", "Bar", "Position"]
        dic["DrumPitch"] = ["DrumPitch", "Program", "Bar", "Position"]

        if self.config.use_chords:
            dic["Program"] += ["Chord"]
            dic["Chord"] = ["Pitch"]

        return dic

    @_in_as_seq()
    def tokens_errors(self, tokens: Union[TokSequence, List, np.ndarray, Any]) -> float:
        r"""Checks if a sequence of tokens is made of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :return: the error ratio (lower is better)
        """
        tokens = tokens.tokens
        err = 0
        previous_type = tokens[0][0].split("_")[0]
        current_pitches = []
        current_bar = int(tokens[0][1].split("_")[1])
        current_pos = tokens[0][2].split("_")[1]
        current_pos = int(current_pos) if current_pos != "None" else -1

        for token in tokens[1:]:
            # debug = {j: self.tokens_to_events([tokens[1:][j]])[0] for j in range(i - 4, min(i + 4, len(tokens[1:])))}
            bar_value = int(token[1].split("_")[1])
            pos_value = token[2].split("_")[1]
            pos_value = int(pos_value) if pos_value != "None" else -1
            token_type, token_value = token[0].split("_")

            if any(tok.split("_")[0] in ["PAD", "MASK"] for i, tok in enumerate(token)):
                err += 1
                continue

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Bar":  # reset
                    current_bar += 1
                    current_pos = -1
                    current_pitches = []
                elif token_type == "Pitch":
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
        return err / len(tokens)
