"""Iterator to be used when training a tokenizer with the 🤗tokenizers library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symusic import Score

from .attribute_controls import create_random_ac_indexes
from .classes import TokSequence
from .constants import SCORE_LOADING_EXCEPTION

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .midi_tokenizer import MusicTokenizer


class TokTrainingIterator:
    r"""
    An iterable class to be used when training a tokenizer.

    It loads music files (MIDI, abc) and tokenize them on the fly, to be used with the
    Hugging Face tokenizers library to build a vocabulary with BPE, Unigram or WordPiece
    models.

    :param tokenizer: tokenizer to use for training.
    :param files_paths: sequence of paths of files to load for training.
    :param tracks_idx_random_ratio_range: range of ratios (between 0 and 1 included) of
        tracks to compute attribute controls on. If ``None`` is given, the attribute
        controls will be computed for all the tracks. (default: ``None``)
    :param bars_idx_random_ratio_range: range of ratios (between 0 and 1 included) of
        bars to compute attribute controls on. If ``None`` is given, the attribute
        controls will be computed for all the bars. (default: ``None``)
    """

    def __init__(
        self,
        tokenizer: MusicTokenizer,
        files_paths: Sequence[Path],
        tracks_idx_random_ratio_range: tuple[float, float] | None = None,
        bars_idx_random_ratio_range: tuple[float, float] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.files_paths = files_paths
        self.tracks_idx_random_ratio_range = (
            tracks_idx_random_ratio_range if tracks_idx_random_ratio_range else []
        )
        self.bars_idx_random_ratio_range = (
            bars_idx_random_ratio_range if bars_idx_random_ratio_range else []
        )
        self.__iter_count = 0

    def load_file(self, path: Path) -> list[str]:
        """
        Load a music file and convert it to its byte representation.

        :param path: path to the file to load.
        :return: the byte representation of the file.
        """
        # Load and tokenize file
        try:
            score = Score(path)
        except SCORE_LOADING_EXCEPTION:
            return []

        # Preprocess first to already have the appropriate tracks idx in case of deletes
        score = self.tokenizer.preprocess_score(score)

        # Randomly create attribute controls indexes
        ac_indexes = None
        if (
            len(self.tracks_idx_random_ratio_range) > 0
            or len(self.bars_idx_random_ratio_range) > 0
        ):
            ac_indexes = create_random_ac_indexes(
                score,
                self.tokenizer.attribute_controls,
                self.tracks_idx_random_ratio_range,
                self.bars_idx_random_ratio_range,
            )

        # Tokenize the file
        # Need to specify `encode_ids=False` as it might be already pretrained
        # For MMM, we make sure to have sequences separated per track
        kwargs = {}
        # can't use isinstance because of circular import
        if type(self.tokenizer).__name__ == "MMM":
            kwargs["concatenate_track_sequences"] = False
        tokseq = self.tokenizer(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            **kwargs,
        )

        # Split ids if requested
        if self.tokenizer.config.encode_ids_split in ["bar", "beat"]:
            if isinstance(tokseq, TokSequence):
                tokseq = [tokseq]

            new_seqs = []
            for seq in tokseq:
                if self.tokenizer.config.encode_ids_split == "bar":
                    new_seqs += seq.split_per_bars()
                else:
                    new_seqs += seq.split_per_beats()
            tokseq = [seq for seq in new_seqs if len(seq) > 0]

        # Convert ids to bytes for training
        if isinstance(tokseq, TokSequence):
            token_ids = tokseq.ids
        else:
            token_ids = [seq.ids for seq in tokseq]
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if isinstance(bytes_, str):
            bytes_ = [bytes_]

        return bytes_

    def __len__(self) -> int:
        """
        Return the number of files in the training corpus.

        :return: number of files in the training corpus.
        """
        return len(self.files_paths)

    def __getitem__(self, idx: int) -> list[str]:
        """
        Convert the ``idx``th file to its byte representation.

        :param idx: idx of the file to convert.
        :return: byte representation of the file.
        """
        return self.load_file(self.files_paths[idx])

    def __iter__(self) -> TokTrainingIterator:  # noqa:D105
        return self

    def __next__(self) -> list[str]:  # noqa:D105
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]

    def __str__(self) -> str:
        """
        Return the ``str`` representation of the iterator.

        :return: string description.
        """
        return f"{self.tokenizer} - {len(self)} files"
