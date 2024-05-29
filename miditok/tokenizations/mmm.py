"""MMM (Multitrack Music Machine) tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import miditok
from miditok import MusicTokenizer
from miditok.classes import Event, TokSequence
from miditok.constants import MMM_COMPATIBLE_TOKENIZERS

if TYPE_CHECKING:
    from pathlib import Path

    from symusic import Score


class MMM(MusicTokenizer):
    r"""
    MMM tokenizer.

    Standing for `Multi-Track Music Machine <https://arxiv.org/abs/2008.06048>`_,
    MMM is a multitrack tokenization primarily designed for music inpainting and
    infilling. Tracks are tokenized independently and concatenated into a single token
    sequence. ``Bar_Fill`` tokens are used to specify the bars to fill (or inpaint, or
    rewrite), the new tokens are then autoregressively generated.
    Note that *this implementation represents note durations with* ``Duration`` *tokens*
    instead of the ``NoteOff`` strategy of the `original paper <https://arxiv.org/abs/2008.06048>`_.
    The reason being that ``NoteOff`` tokens perform poorer for generation with causal
    models.

    **Add a** ``density_bins_max`` **entry in the config, mapping to a tuple specifying
    the number of density bins, and the maximum density in notes per beat to consider.
    (default: (10, 20))**

    **Note:** When decoding tokens with tempos, only the tempos of the first track
    will be decoded.
    """

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_programs = True
        self.config.program_changes = True
        self.config.one_token_stream_for_programs = True

        self.concatenate_track_sequences = True

        # Checks base tokenizer argument
        if "base_tokenizer" not in self.config.additional_params:
            msg = (
                "MMM must be used with a `base_tokenizer`. This argument must be set in"
                " `config.additional_params` and reference to one of "
                f"{MMM_COMPATIBLE_TOKENIZERS}."
            )
            raise ValueError(msg)
        tokenizer_name = self.config.additional_params["base_tokenizer"]
        if tokenizer_name not in MMM_COMPATIBLE_TOKENIZERS:
            msg = (
                '`config.additional_params["base_tokenizer"]` must be one of '
                f"{MMM_COMPATIBLE_TOKENIZERS}, received {tokenizer_name}."
            )
            raise ValueError(msg)

        # Add Track_Start and Track_End tokens to config
        for token in ("Track_Start", "Track_End"):
            if token not in self.config.special_tokens:
                self.config.special_tokens.append(token)

        # Create base tokenizer
        base_tokenizer_config = self.config.copy()
        base_tokenizer_config.one_token_stream_for_programs = False
        self.base_tokenizer = getattr(miditok, tokenizer_name)(base_tokenizer_config)
        self.base_tokenizer.config.use_programs = True
        self._note_on_off = self.base_tokenizer._note_on_off

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
        if len(events) < 2:  # empty list
            track_events = events.copy()
        else:
            track_events = self.base_tokenizer._add_time_events(events, time_division)

        track_start_event = Event("Track", "Start", 0)
        track_end_event = Event("Track", "End", track_events[-1].time + 1)
        return [track_start_event, *track_events, track_end_event]

    def encode(
        self,
        score: Score | Path,
        encode_ids: bool = True,
        no_preprocess_score: bool = False,
        concatenate_track_sequences: bool = True,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Tokenize a music file (MIDI/abc), given as a ``symusic.Score`` or a file path.

        You can provide a ``Path`` to the file to tokenize, or a ``symusic.Score``
        object.
        This method returns a (list of) :class:`miditok.TokSequence`.

        If you are implementing your own tokenization by subclassing this class,
        **override the protected** ``_score_to_tokens`` **method**.

        :param score: the ``symusic.Score`` object to convert.
        :param encode_ids: the backbone model (BPE, Unigram, WordPiece) will encode the
            tokens and compress the sequence. Can only be used if the tokenizer has been
            trained. (default: ``True``)
        :param no_preprocess_score: whether to preprocess the ``symusic.Score``. If this
            argument is provided as ``True``, make sure that the corresponding music
            file / ``symusic.Score`` has already been preprocessed by the tokenizer
            (:py:func:`miditok.MusicTokenizer.preprocess_score`) or that its content is
            aligned with the tokenizer's vocabulary, otherwise the tokenization is
            likely to crash. This argument is useful in cases where you need to use the
            preprocessed ``symusic.Score`` along with the tokens to not have to
            preprocess it twice as this method preprocesses it inplace.
            (default: ``False``)
        :param concatenate_track_sequences: will concatenate the token sequences of each
            track after tokenizing them. (default: ``True``)
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        self.concatenate_track_sequences = concatenate_track_sequences
        return super().encode(score, encode_ids, no_preprocess_score)

    def _score_to_tokens(self, score: Score) -> TokSequence | list[TokSequence]:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        We need to override the parent method to concatenate the tracks sequences.

        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If `one_token_stream` is
        ``True``, all events of all tracks are treated all at once, otherwise the
        events of each track are treated independently.

        :param score: the :class:`symusic.Score` object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        self.one_token_stream = False
        sequences = super()._score_to_tokens(score)
        self.one_token_stream = True

        # Concatenate the sequences
        if self.concatenate_track_sequences:
            return sum(sequences)
        return sequences

    def _sort_events(self, events: list[Event]) -> None:
        self.base_tokenizer._sort_events(events)

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
        tokseqs = []
        i = 0
        while i < len(tokens):
            if tokens.tokens[i] != "Track_Start":
                i += 1
                continue

            j = i
            while j < len(tokens) and tokens.tokens[j] != "Track_End":
                j += 1

            # Extract the subseq
            tokseqs.append(tokens[i + 1 : j])
            i = j

        return self.base_tokenizer._tokens_to_score(tokseqs)

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
        # `_tweak_config_before_creating_voc` already called, this method returns the
        # vocab of the base_tokenizer, without the special tokens as they will be
        # re-added by the `__create_vocabulary` method.
        base_voc = list(self.base_tokenizer.vocab.keys())
        for special_token in self.config.special_tokens:
            base_voc.remove(special_token)
        return base_voc

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        return self.base_tokenizer.tokens_types_graph.copy()

    def _tokens_errors(self, tokens: list[str]) -> int:
        """
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
        err = 0
        i = 0
        while i < len(tokens):
            if tokens[i] != "Track_Start":
                i += 1
                continue

            j = i
            while j < len(tokens) and tokens[j] != "Track_End":
                j += 1

            # Compute err for base_tokenizer
            err += self.base_tokenizer._tokens_errors(tokens[i + 1 : j])
            i = j
        return err
