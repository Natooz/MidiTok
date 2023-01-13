"""Octuple encoding method, as introduced in MusicBERT
https://arxiv.org/abs/2106.05630

"""

from math import ceil
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event
from .constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    TIME_DIVISION,
    TIME_SIGNATURE,
    TEMPO,
    MIDI_INSTRUMENTS,
)


class Octuple(MIDITokenizer):
    r"""Octuple encoding method, as introduced in MusicBERT
    https://arxiv.org/abs/2106.05630

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (time signature, tempo)
    :param pad: will include a PAD token, used when training a model with batch of sequences of
            unequal lengths, and usually at index 0 of the vocabulary. (default: True)
    :param sos_eos: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary.
            (default: False)
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
        programs: List[int] = None,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        # used in place of positional encoding
        self.programs = list(range(-1, 128)) if programs is None else programs
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        token_types = ["Pitch", "Velocity", "Duration", "Program", "Position", "Bar"]
        if additional_tokens["Tempo"]:
            token_types.append("Tempo")
        if additional_tokens["TimeSignature"]:
            token_types.append("TimeSignature")
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used for data augmentation
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            pad,
            sos_eos,
            mask,
            True,
            params=params,
        )

    def save_params(
        self, out_path: Union[str, Path, PurePath], additional_attributes: Dict = None
    ):
        r"""Overrides the parent class method to include programs and max_bar_embedding.
        Saves the config / base parameters of the tokenizer in a file.
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy.
        NOTE: if you override this method, you should probably call it (super()) at the end
            and use the additional_attributes argument.
        NOTE 2: as json cant save tuples as keys, the beat ranges are saved as strings
        with the form startingBeat_endingBeat (underscore separating these two values)

        :param out_path: output path to save the file
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
        """
        if additional_attributes is None:
            additional_attributes = {}
        additional_attributes_tmp = {
            "max_bar_embedding": self.max_bar_embedding,
            "programs": self.programs,
            **additional_attributes,
        }
        super().save_params(out_path, additional_attributes_tmp)

    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[int]]:
        r"""Override the parent class method
        Converts a MIDI file in a tokens representation, a sequence of "time steps".
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)

        :param midi: the MIDI objet to convert
        :return: the token representation, i.e. tracks converted into sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self.durations_ticks:
            self.durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self.durations
                ]
            )

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self.current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        # **************** OVERRIDE FROM HERE, KEEP THE LINES ABOVE IN YOUR METHOD ****************

        # Check bar embedding limit, update if needed
        nb_bars = ceil(midi.max_tick / (midi.ticks_per_beat * 4))
        if self.max_bar_embedding < nb_bars:
            self.vocab[5].add_event(
                f"Bar_{i}" for i in range(self.max_bar_embedding, nb_bars)
            )
            self.max_bar_embedding = nb_bars

        # Convert each track to tokens
        tokens = []
        for track in midi.instruments:
            if track.program in self.programs:
                tokens += self.track_to_tokens(track)

        tokens.sort(
            key=lambda x: (x[0].time, x[0].desc, x[0].value)
        )  # Sort by time then track then pitch

        # Convert pitch events into tokens
        for time_step in tokens:
            time_step[0] = self.vocab[0].event_to_token[
                f"{time_step[0].type}_{time_step[0].value}"
            ]

        return tokens

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, int]]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch (as an Event object for sorting purpose afterwards)
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)

        :param track: track object to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self.current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.beat_res.values())
        dur_bins = self.durations_ticks[time_division]

        events = []
        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        current_time_sig_idx = 0
        current_time_sig_tick = 0
        current_time_sig_bar = 0
        time_sig_change = self.current_midi_metadata["time_sig_changes"][
            current_time_sig_idx
        ]
        current_time_sig = self._reduce_time_signature(
            time_sig_change.numerator, time_sig_change.denominator
        )
        ticks_per_bar = time_division * current_time_sig[0]

        for note in track.notes:
            # Positions and bars
            if note.start != current_tick:
                pos_index = int(
                    ((note.start - current_time_sig_tick) % ticks_per_bar)
                    / ticks_per_sample
                )
                current_tick = note.start
                current_bar = (
                    current_time_sig_bar
                    + (current_tick - current_time_sig_tick) // ticks_per_bar
                )
                current_pos = pos_index

            # Note attributes
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            event = [
                Event(
                    type_="Pitch",
                    value=note.pitch,
                    time=note.start,
                    desc=-1 if track.is_drum else track.program,
                ),
                self.vocab[1].event_to_token[f"Velocity_{note.velocity}"],
                self.vocab[2].event_to_token[
                    f'Duration_{".".join(map(str, self.durations[dur_index]))}'
                ],
                self.vocab[3].event_to_token[
                    f"Program_{-1 if track.is_drum else track.program}"
                ],
                self.vocab[4].event_to_token[f"Position_{current_pos}"],
                self.vocab[5].event_to_token[f"Bar_{current_bar}"],
            ]

            # (Tempo)
            if self.additional_tokens["Tempo"]:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(
                    self.current_midi_metadata["tempo_changes"]
                ):
                    # Will loop over incoming tempo changes
                    for tempo_change in self.current_midi_metadata["tempo_changes"][current_tempo_idx + 1:]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note.start:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += (
                                1  # update tempo value (might not change) and index
                            )
                        elif tempo_change.time > note.start:
                            break  # this tempo change is beyond the current time step, we break the loop
                event.append(self.vocab[6].event_to_token[f"Tempo_{current_tempo}"])

            # (TimeSignature)
            if self.additional_tokens["TimeSignature"]:
                # If the current time signature is not the last one
                if current_time_sig_idx + 1 < len(
                    self.current_midi_metadata["time_sig_changes"]
                ):
                    # Will loop over incoming time signature changes
                    for time_sig_change in self.current_midi_metadata[
                        "time_sig_changes"
                    ][current_time_sig_idx + 1:]:
                        # If this time signature change happened before the current moment
                        if time_sig_change.time <= note.start:
                            current_time_sig = self._reduce_time_signature(
                                time_sig_change.numerator, time_sig_change.denominator
                            )
                            current_time_sig_idx += 1  # update time signature value (might not change) and index
                            current_time_sig_bar += (
                                time_sig_change.time - current_time_sig_tick
                            ) // ticks_per_bar
                            current_time_sig_tick = time_sig_change.time
                            ticks_per_bar = time_division * current_time_sig[0]
                        elif time_sig_change.time > note.start:
                            break  # this time signature change is beyond the current time step, we break the loop
                event.append(
                    self.vocab[-1].event_to_token[
                        f"TimeSig_{current_time_sig[0]}/{current_time_sig[1]}"
                    ]
                )

            events.append(event)

        return events

    def tokens_to_midi(
        self,
        tokens: List[List[int]],
        _=None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)
        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param _: unused, to match parent method signature
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert (
            time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // max(self.beat_res.values())
        events = self.tokens_to_events(tokens)

        tempo_changes = [TempoChange(TEMPO, 0)]
        if self.additional_tokens["Tempo"]:
            for i in range(len(events)):
                if events[i][6].value != "None":
                    tempo_changes = [TempoChange(int(events[i][6].value), 0)]
                    break

        time_sig = TIME_SIGNATURE
        if self.additional_tokens["TimeSignature"]:
            for i in range(len(events)):
                if events[i][-1].value != "None":
                    time_sig = self._parse_token_time_signature(events[i][-1].value)
                    break

        ticks_per_bar = time_division * time_sig[0]
        time_sig_changes = [TimeSignature(*time_sig, 0)]

        current_time_sig_tick = 0
        current_time_sig_bar = 0

        tracks = dict([(n, []) for n in range(-1, 128)])
        for time_step in events:
            if any(tok.value == "None" for tok in time_step[:6]):
                continue  # Either padding, mask: error of prediction or end of sequence anyway

            # Note attributes
            pitch = int(time_step[0].value)
            vel = int(time_step[1].value)
            duration = self._token_duration_to_ticks(time_step[2].value, time_division)

            # Time and track values
            program = int(time_step[3].value)
            current_pos = int(time_step[4].value)
            current_bar = int(time_step[5].value)
            current_tick = (
                current_time_sig_tick
                + (current_bar - current_time_sig_bar) * ticks_per_bar
                + current_pos * ticks_per_sample
            )

            # Append the created note
            tracks[program].append(
                Note(vel, pitch, current_tick, current_tick + duration)
            )

            # Tempo, adds a TempoChange if necessary
            if self.additional_tokens["Tempo"] and time_step[6].value != "None":
                tempo = int(time_step[6].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))

            # Time Signature, adds a TimeSignatureChange if necessary
            if (
                self.additional_tokens["TimeSignature"]
                and time_step[-1].value != "None"
            ):
                time_sig = self._parse_token_time_signature(time_step[-1].value)
                if time_sig != (
                    time_sig_changes[-1].numerator,
                    time_sig_changes[-1].denominator,
                ):
                    current_time_sig_tick += (
                        current_bar - current_time_sig_bar
                    ) * ticks_per_bar
                    current_time_sig_bar = current_bar
                    ticks_per_bar = time_division * time_sig[0]
                    time_sig_changes.append(
                        TimeSignature(*time_sig, current_time_sig_tick)
                    )

        # Tempos
        midi.tempo_changes = tempo_changes

        # Time Signatures
        midi.time_signature_changes = time_sig_changes

        # Appends created notes to MIDI object
        for program, notes in tracks.items():
            if len(notes) == 0:
                continue
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
        tokens: List[List[int]],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""NOT RELEVANT / IMPLEMENTED IN OCTUPLE
        Use tokens_to_midi instead

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        raise NotImplementedError(
            "tokens_to_track not implemented for Octuple, use tokens_to_midi instead"
        )

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
            for _ in range(6)
        ]

        # PITCH
        vocab[0].add_event(f"Pitch_{i}" for i in self.pitch_range)

        # VELOCITY
        vocab[1].add_event(f"Velocity_{i}" for i in self.velocities)

        # DURATION
        vocab[2].add_event(
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        )

        # PROGRAM
        vocab[3].add_event(f"Program_{i}" for i in self.programs)

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab[4].add_event(f"Position_{i}" for i in range(nb_positions))

        # BAR
        vocab[5].add_event(
            f"Bar_{i}" for i in range(self.max_bar_embedding)
        )  # bar embeddings (positional encoding)

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event(f"Tempo_{i}" for i in self.tempos)

        # TIME_SIGNATURE
        if self.additional_tokens["TimeSignature"]:
            vocab.append(
                Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
            )
            vocab[-1].add_event(f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Not relevant for Octuple.

        :return: the token types transitions dictionary
        """
        return {}  # not relevant for this encoding

    def token_types_errors(
        self, tokens: List[List[int]], consider_pad: bool = False
    ) -> float:
        r"""Checks if a sequence of tokens is constituted of good token values and
        returns the error ratio (lower is better).
        The token types are always the same in Octuple so this methods only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err = 0
        current_bar = current_pos = -1
        current_pitches = {p: [] for p in self.programs}

        for token in tokens:
            if consider_pad and all(
                token[i] == self.vocab[i]["PAD_None"] for i in range(len(token))
            ):
                break
            if any(
                self.vocab[i][token].split("_")[1] == "None"
                for i, token in enumerate(token)
            ):
                err += 1
                continue
            has_error = False
            bar_value = int(self.vocab[5].token_to_event[token[5]].split("_")[1])
            pos_value = int(self.vocab[4].token_to_event[token[4]].split("_")[1])
            pitch_value = int(self.vocab[0].token_to_event[token[0]].split("_")[1])
            program_value = int(self.vocab[3].token_to_event[token[3]].split("_")[1])

            # Bar
            if bar_value < current_bar:
                has_error = True
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = -1
                current_pitches = {p: [] for p in self.programs}

            # Position
            if pos_value < current_pos:
                has_error = True
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = {p: [] for p in self.programs}

            # Pitch
            if pitch_value in current_pitches[program_value]:
                has_error = True
            else:
                current_pitches[program_value].append(pitch_value)

            if has_error:
                err += 1

        return err / len(tokens)
