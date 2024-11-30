"""MMM (Multitrack Music Machine) tokenizer."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

import miditok
from miditok import MusicTokenizer
from miditok.classes import Event, TokSequence
from miditok.constants import MMM_COMPATIBLE_TOKENIZERS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from symusic import Score

    from miditok import TokenizerConfig


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

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.TokenizerConfig` object.
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful if
        the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path | None = None,
    ) -> None:
        # Directly call super method
        # __in_init is used to not call the `add_to_vocab` method for the base_tokenizer
        # when creating the vocab of the MMM object. The MMM
        self.__in_init = True
        super().__init__(tokenizer_config, params)
        # Set to True, whereas `config.one_token_stream_for_programs` is False
        self.one_token_stream = True
        self.__in_init = False
        # We don't need to specifically load the base_tokenizer from the config file as
        # it can be entirely created from the `self` config file only and will only be
        # used for the `_add_time_events`, `_sort_events`, `_tokens_to_score`,
        # `_tokens_errors` and mirrored base vocabulary (created from config).

    def _tweak_config_before_creating_voc(self) -> None:
        # The Programs are specified at the beginning of each track token sequence.
        self.config.use_programs = True
        self.config.program_changes = True
        # one_token_stream_for_programs is False so that the base_tokenizer treats each
        # track independently ((I,T) io) but one_token_stream True (set in __init__)
        # so that self (MMM)
        # has a (T) io as it will concatenate the tracks token sequences.
        self.config.one_token_stream_for_programs = False

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
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
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
        :param attribute_controls_indexes: indices of the attribute controls to compute
            and associated tracks and bars. This argument has to be provided as a
            dictionary mapping track indices to dictionaries mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of bar
            indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as:
            ``{track_idx: {ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls.
        :param concatenate_track_sequences: will concatenate the token sequences of each
            track after tokenizing them. (default: ``True``)
        :return: a :class:`miditok.TokSequence` if ``concatenate_track_sequences`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Need to override to set this class attribute that will be used in
        # `_score_to_tokens`.
        sequences = super().encode(
            score,
            encode_ids,
            no_preprocess_score,
            attribute_controls_indexes,
        )
        # Concatenate the sequences
        if concatenate_track_sequences:
            return sum(sequences)
        return sequences

    def encode_token_ids(self, seq: TokSequence | list[TokSequence]) -> None:
        """
        Encode a :class:`miditok.TokSequence` with BPE, Unigram or WordPiece.

        The method works inplace and only alters the sequence's ``.ids``.
        The method also works with lists of :class:`miditok.TokSequence`.
        If a list is given, the model will encode all sequences in one batch to speed up
        the operation.

        :param seq: :class:`miditok.TokSequence` to encode ids.
        """
        # If no split --> just call parent method
        if self.config.encode_ids_split == "no":
            super().encode_token_ids(seq)

        # If `seq` is not a TokSequence (so a list) --> recursively call this method
        elif isinstance(seq, list):
            for subseq in seq:
                self.encode_token_ids(subseq)

        # Otherwise (it's a TokSequence) we must make sure to split the ids per
        # bars/beats for each separate track: splitting the TokSequence as the parent
        # method doesn't handle concatenated tracks in one single token sequence.
        else:
            seqs_split = self.split_tokseq_per_track(
                deepcopy(seq), keep_track_tokens=True
            )
            super().encode_token_ids(seqs_split)

            # Concatenate ids of the split sequences now encoded
            seq.ids = []
            for subseq in seqs_split:
                seq.ids += subseq.ids
            seq.are_ids_encoded = True

    def _sort_events(self, events: list[Event]) -> None:
        self.base_tokenizer._sort_events(events)

    def split_tokseq_per_track(
        self,
        tokseq: TokSequence,
        keep_track_tokens: bool = False,
    ) -> list[TokSequence]:
        """
        Split an MMM :class:`miditok.TokSequence` per tracks.

        :param tokseq: :class:`miditok.TokSequence` token sequence.
        :param keep_track_tokens: whether to keep the ``Track_Start/End`` tokens.
            (default: ``False``)
        :return: list :class:`miditok.TokSequence`, one for each track in ``tokseq``.
        """
        track_tokens_idx = np.where(np.array(tokseq.ids) == self.vocab["Track_Start"])[
            0
        ].tolist()
        if len(track_tokens_idx) == 0:
            return [tokseq]

        tokseqs = []
        for i, track_idx in enumerate(track_tokens_idx):
            if i + 1 == len(track_tokens_idx):
                idx_end = None
            elif keep_track_tokens:
                idx_end = track_tokens_idx[i + 1]
            else:
                idx_end = track_tokens_idx[i + 1] - 1
            idx_start = track_idx if keep_track_tokens else track_idx + 1
            tokseqs.append(tokseq[idx_start:idx_end])

        return tokseqs

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
        tokseqs = self.split_tokseq_per_track(tokens)
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

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        :return: the vocabulary as a list of string.
        """
        return self.base_tokenizer._create_base_vocabulary()

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

    def add_to_vocab(
        self,
        token: str | Event,
        special_token: bool = False,
        vocab_idx: int | None = None,
        byte_: str | None = None,
    ) -> None:
        r"""
        Add an event to the vocabulary. Its id will be the length of the vocab.

        :param token: token to add, as a formatted string of the form "Type_Value",
            e.g. Pitch_80, or an Event.
        :param special_token: whether the token is special. (default: ``False``)
        :param vocab_idx: idx of the vocabulary (in case of embedding pooling).
            (default: ``None``)
        :param byte_: unique byte associated to the token. The associated byte of a
            token is used to encode-decode ids with the tokenizer's model (BPE, Unigram,
            WordPiece). If None is given, it will default to ``chr(id_ + CHR_ID_START)``
            . (default: ``None``)
        """
        # Overridden to make sure the vocabs of self and base_tokenizer remain identical
        if not self.__in_init:
            self.base_tokenizer.add_to_vocab(
                token,
                special_token,
                vocab_idx,
                byte_,
            )
        super().add_to_vocab(token, special_token, vocab_idx, byte_)
