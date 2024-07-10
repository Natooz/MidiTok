"""Dataset classes to be used with PyTorch when training a model."""

from __future__ import annotations

import json
from abc import ABC
from typing import TYPE_CHECKING, Any

from symusic import Score
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok.attribute_controls import create_random_ac_indexes
from miditok.constants import SCORE_LOADING_EXCEPTION

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from miditok import MusicTokenizer, TokSequence


class _DatasetABC(Dataset, ABC):
    r"""
    Abstract ``Dataset`` class.

    It holds samples (and optionally labels) and implements the basic magic methods.
    """

    def __init__(
        self,
    ) -> None:
        self.__iter_count = 0

    @staticmethod
    def _preprocess_token_ids(
        token_ids: list[int | list[int]],
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        enforce_eos_token_if_seq_len_exceed_lim: bool = False,
    ) -> list[int | list[int]]:
        if len(token_ids) == 0:
            return token_ids

        # Reduce sequence length
        max_seq_len -= sum([1 for t in [bos_token_id, eos_token_id] if t is not None])
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            if not enforce_eos_token_if_seq_len_exceed_lim:
                eos_token_id = None

        # Adds BOS and EOS tokens
        if bos_token_id:
            if isinstance(token_ids[0], list):
                bos_token_id = [bos_token_id] * len(token_ids[0])
            token_ids.insert(0, bos_token_id)
        if eos_token_id:
            if isinstance(token_ids[0], list):
                eos_token_id = [eos_token_id] * len(token_ids[0])
            token_ids.append(eos_token_id)

        return token_ids

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def __iter__(self) -> _DatasetABC:
        return self

    def __next__(self) -> Mapping[str, Any]:
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]


class DatasetMIDI(_DatasetABC):
    r"""
    A ``Dataset`` loading and tokenizing music files (MIDI, abc) during training.

    This class can be used for either tokenize music files on the fly when iterating it,
    or by pre-tokenizing all the files at its initialization and store the tokens in
    memory.

    **Important note:** you should probably use this class in concert with the
    :py:func:`miditok.pytorch_data.split_files_for_training` method in order to train
    your model with chunks of music files having token sequence lengths close to the
    ``max_seq_len`` value.
    When using this class with file chunks, the ``BOS`` and ``EOS`` tokens will only be
    added to the first and last chunks respectively. This allows to not train the model
    with ``EOS`` tokens that would incorrectly inform the model the end of the data
    samples, and break the causality chain of consecutive chunks with incorrectly
    placed ``BOS`` tokens.

    Additionally, you can use the ``func_to_get_labels`` argument to provide a method
    allowing to use labels (one label per file).

    **Handling of corrupted files:**
    Some MIDI files may be corrupted, as for example containing unexpected values.
    In such cases, if the ``DatasetMIDI`` pre-tokenizes, it will simply ignore these
    files. Otherwise, the ``DatasetMIDI`` will return dictionaries with ``None`` values
    when iterated.

    :param files_paths: paths to the music files to load.
    :param tokenizer: tokenizer.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    :param pre_tokenize: whether to pre-tokenize the data files when creating the
        ``Dataset`` object. If this is enabled, the ``Dataset`` will tokenize all the
        files at its initialization and store the tokens in memory.
    :param ac_tracks_random_ratio_range: range of ratios (between 0 and 1 included) of
        tracks to compute attribute controls on. If ``None`` is given, no track
        attribute control will be used. (default: ``None``)
    :param ac_bars_random_ratio_range: range of ratios (between 0 and 1 included) of
        bars to compute attribute controls on. If ``None`` is given, no bar attribute
        control will be used. (default: ``None``)
    :param func_to_get_labels: a function to retrieve the label of a file. The method
        must take two positional arguments: the first is either the
        :class:`miditok.TokSequence` returned when tokenizing a file, the second is the
        path to the file just loaded. The method must return an integer which
        corresponds to the label id (and not the absolute value, e.g. if you are
        classifying 10 musicians, return the id from 0 to 9 included corresponding to
        the musician). (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        tokenizer: MusicTokenizer,
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pre_tokenize: bool = False,
        ac_tracks_random_ratio_range: tuple[float, float] | None = None,
        ac_bars_random_ratio_range: tuple[float, float] | None = None,
        func_to_get_labels: Callable[
            [Score, TokSequence | list[TokSequence], Path],
            int | list[int] | LongTensor,
        ]
        | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        super().__init__()

        # Set class attributes
        self.files_paths = list(files_paths).copy()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pre_tokenize = pre_tokenize
        self.tracks_idx_random_ratio_range = ac_tracks_random_ratio_range
        self.bars_idx_random_ratio_range = ac_bars_random_ratio_range
        self.func_to_get_labels = func_to_get_labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        self.samples, self.labels = [], [] if func_to_get_labels else None

        # Pre-tokenize the files
        if pre_tokenize:
            for file_path in tqdm(
                self.files_paths,
                desc="Pre-tokenizing",
                miniters=int(len(self.files_paths) / 20),
                maxinterval=480,
            ):
                try:
                    score = Score(file_path)
                except SCORE_LOADING_EXCEPTION:
                    continue
                tokseq = self._tokenize_score(score)
                if tokenizer.one_token_stream:
                    tokseq = [tokseq]
                for seq in tokseq:
                    self.samples.append(LongTensor(seq.ids))
                    if func_to_get_labels:
                        label = func_to_get_labels(score, seq, file_path)
                        if not isinstance(label, LongTensor):
                            label = LongTensor(label)
                        self.labels.append(label)

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Return the ``idx`` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the ``idx``th file on the fly. If the file to is
        corrupted, the method will return a dictionary with ``None`` values.

        :param idx: idx of the file/sample.
        :return: the token ids, with optionally the associated label.
        """
        labels = None

        # Already pre-tokenized
        if self.pre_tokenize:
            token_ids = self.samples[idx]
            if self.func_to_get_labels is not None:
                labels = self.labels[idx]

        # Tokenize on the fly
        else:
            # The tokenization steps are outside the try bloc as if there are errors,
            # we might want to catch them to fix them instead of skipping the iteration.
            try:
                score = Score(self.files_paths[idx])
            except SCORE_LOADING_EXCEPTION:
                item = {self.sample_key_name: None}
                if self.func_to_get_labels is not None:
                    item[self.labels_key_name] = labels
                return item

            tseq = self._tokenize_score(score)
            # If not one_token_stream, we only take the first track/sequence
            token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids
            if self.func_to_get_labels is not None:
                # tokseq can be given as a list of TokSequence to get the labels
                labels = self.func_to_get_labels(score, tseq, self.files_paths[idx])
                if not isinstance(labels, LongTensor):
                    labels = LongTensor([labels] if isinstance(labels, int) else labels)

        item = {self.sample_key_name: LongTensor(token_ids)}
        if self.func_to_get_labels is not None:
            item[self.labels_key_name] = labels

        return item

    def _tokenize_score(self, score: Score) -> TokSequence | list[TokSequence]:
        # Preprocess the music file
        score = self.tokenizer.preprocess_score(score)

        # Randomly create attribute controls indexes
        ac_indexes = None
        if self.tracks_idx_random_ratio_range or self.bars_idx_random_ratio_range:
            ac_indexes = create_random_ac_indexes(
                score,
                self.tokenizer.attribute_controls,
                self.tracks_idx_random_ratio_range,
                self.bars_idx_random_ratio_range,
            )

        # Tokenize it
        tokseq = self.tokenizer.encode(
            score,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
        )

        # If tokenizing on the fly a multi-stream tokenizer, only keeps the first track
        if not self.pre_tokenize and not self.tokenizer.one_token_stream:
            tokseq = [tokseq[0]]

        # If this file is a chunk (split_files_for_training), determine its id.
        # By default, we add BOS and EOS tokens following the values of
        # self.bos_token_id and self.eos_token_id (that may be None), except when the
        # file is identified as a chunk.
        add_bos_token = add_eos_token = True
        for marker in score.markers:
            if marker.time != 0:
                break
            if marker.text.startswith("miditok: chunk"):
                chunk_id, chunk_id_last = map(
                    int, marker.text.split(" ")[-1].split("/")
                )
                add_bos_token = chunk_id == 0
                add_eos_token = chunk_id == chunk_id_last

        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens
        if self.tokenizer.one_token_stream:
            tokseq.ids = self._preprocess_token_ids(
                tokseq.ids,
                self.max_seq_len,
                self.bos_token_id if add_bos_token else None,
                self.eos_token_id if add_eos_token else None,
                enforce_eos_token_if_seq_len_exceed_lim=False,
            )
        else:
            for seq in tokseq:
                seq.ids = self._preprocess_token_ids(
                    seq.ids,
                    self.max_seq_len,
                    self.bos_token_id if add_bos_token else None,
                    self.eos_token_id if add_eos_token else None,
                    enforce_eos_token_if_seq_len_exceed_lim=False,
                )

        return tokseq

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return len(self.samples) if self.pre_tokenize else len(self.files_paths)

    def __repr__(self) -> str:  # noqa:D105
        return self.__str__()

    def __str__(self) -> str:  # noqa:D105
        if self.pre_tokenize:
            return f"Pre-tokenized dataset with {len(self.samples)} samples"
        return f"{len(self.files_paths)} files."


class DatasetJSON(_DatasetABC):
    r"""
    Basic ``Dataset`` loading JSON files of tokenized music files.

    When indexed (``dataset[idx]``), a ``DatasetJSON`` will load the
    ``files_paths[idx]`` JSON file and return the token ids, that can be used to train
    generative models.

    **This class is only compatible with tokens saved as a single stream of tokens
    (** ``tokenizer.one_token_stream`` **).** If you plan to use it with token files
    containing multiple token streams, you should first split each track token sequence
    with the :py:func:`miditok.pytorch_data.split_dataset_to_subsequences` method.

    If your dataset contains token sequences with lengths largely varying, you might
    want to first split it into subsequences with the
    :py:func:`miditok.pytorch_data.split_files_for_training` method before loading
    it to avoid losing data.

    :param files_paths: list of paths to files to load.
    :param max_seq_len: maximum sequence length (in num of tokens). (default: ``None``)
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.files_paths = files_paths
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._effective_max_seq_len = max_seq_len - sum(
            [1 for tok in [bos_token_id, eos_token_id] if tok is not None]
        )
        super().__init__()

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Load the tokens from the ``idx`` JSON file.

        :param idx: index of the file to load.
        :return: the tokens as a dictionary mapping to the token ids as a tensor.
        """
        with self.files_paths[idx].open() as json_file:
            token_ids = json.load(json_file)["ids"]
        token_ids = self._preprocess_token_ids(
            token_ids,
            self._effective_max_seq_len,
            self.bos_token_id,
            self.eos_token_id,
        )

        return {"input_ids": LongTensor(token_ids)}

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return len(self.files_paths)
