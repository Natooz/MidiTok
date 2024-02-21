"""Dataset classes to be used with PyTorch when training a model."""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from symusic import Score
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok.utils import split_midi

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from miditok import MIDITokenizer


class _DatasetABC(Dataset, ABC):
    r"""
    Abstract ``Dataset`` class.

    It holds samples (and optionally labels) and implements the basic magic methods.
    """

    def __init__(
        self,
    ) -> None:
        self.__iter_count = 0

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


class DatasetTok(_DatasetABC):
    r"""
    Basic ``Dataset`` loading and tokenizing MIDIs or JSON token files.

    The token ids will be stored in RAM. It outputs token sequences that can be used to
    train models.

    The tokens sequences being loaded will then be split into subsequences, of length
    comprise between ``min_seq_len`` and ``max_seq_len``.
    For example, with ``min_seq_len = 50`` and ``max_seq_len = 100``:
    * a sequence of 650 tokens will be split into 6 subsequences of 100 tokens plus one
    subsequence of 50 tokens;
    * a sequence of 620 tokens will be split into 6 subsequences of 100 tokens, the
    last 20 tokens will be discarded;
    * a sequence of 670 tokens will be split into 6 subsequences of 100 tokens plus one
    subsequence of 50 tokens, and the last 20 tokens will be discarded.

    This `Dataset` class is well suited if you have enough RAM to store all the data,
    as it does not require you to prior split the dataset into subsequences of the
    length you desire. Note that if you directly load MIDI files, the loading can take
    some time as they will need to be tokenized. You might want to tokenize them before
    once with the ``tokenizer.tokenize_midi_dataset()`` method.

    Additionally, you can use the `func_to_get_labels` argument to provide a method
    allowing to use labels (one label per file).

    :param files_paths: paths to MIDI files to load.
    :param tokenizer: tokenizer.
    :param min_seq_len: minimum sequence length (in num of tokens)
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param midi_split_max_num_beat:
    :param midi_split_min_num_beat:
    :param pre_tokenize:
    :param save_dir:
    :param save_dir_tmp:
    :param func_to_get_labels: a function to retrieve the label of a file. The method
        must take two positional arguments: the first is either a MidiFile or the
        tokens loaded from the json file, the second is the path to the file just
        loaded. The method must return an integer which correspond to the label id
        (and not the absolute value, e.g. if you are classifying 10 musicians, return
        the id from 0 to 9 included corresponding to the musician). (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        tokenizer: MIDITokenizer,
        min_seq_len: int,
        max_seq_len: int,
        midi_split_max_num_beat: int | None = None,
        midi_split_min_num_beat: int = 1,
        pre_tokenize: bool = False,
        save_dir: Path | None = None,
        save_dir_tmp: bool = False,
        func_to_get_labels: Callable[[Score | Sequence, Path], int] | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        super().__init__()
        # If MIDI splitting, check the save_dir doesn't already contain them
        if midi_split_max_num_beat and save_dir.is_dir():
            # TODO find a way to identify if save_dir contains split MIDIs
            files_paths = list(save_dir.glob("**/*.mid"))
            # Disable split, no need anymore
            midi_split_max_num_beat = None

        # Set class attributes
        self.files_paths = files_paths
        self.tokenizer = tokenizer
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pre_tokenize = pre_tokenize
        self.func_to_get_labels = func_to_get_labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        if save_dir_tmp:
            save_dir = Path(mkdtemp(prefix="miditok-dataset"))
        self.samples, self.labels = ([], []) if func_to_get_labels else (None, None)

        # TODO method to help to determine midi_split_max_num_beat

        # Load the MIDIs here to split and/or pre-tokenize them
        if midi_split_max_num_beat or pre_tokenize:
            for file_path in tqdm(
                files_paths,
                desc=f"Preprocessing files: {files_paths[0].parent}",
                miniters=int(len(files_paths) / 20),
                maxinterval=480,
            ):
                midis = Score(file_path)

                if midi_split_max_num_beat:
                    midis_split = split_midi(
                        midis, midi_split_max_num_beat, midi_split_min_num_beat
                    )
                    # Save them if needed
                    if save_dir:
                        for _i, midi in enumerate(midis_split):
                            midi.dump(save_dir / "")  # TODO good file tree
                else:
                    midis = [midis]

                # Pre-tokenize the MIDI(s) and store the tokens in memory
                if pre_tokenize:
                    for midi in midis:
                        token_ids = tokenizer.midi_to_tokens(midi).ids
                        # TODO document that this is only possible when pre-tokenizing
                        if tokenizer.one_token_stream:
                            token_ids = [token_ids]
                        for seq in token_ids:
                            self.samples.append(LongTensor(seq))
                            if func_to_get_labels:
                                self.labels.append(func_to_get_labels(seq, file_path))

        if self.labels:
            self.labels = LongTensor(self.labels)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        """
        Return the `idx` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the `idx`th MIDI file on the fly.

        :param idx: idx of the file/sample.
        :return: the token ids, with optionally the associated label.
        """
        labels = None
        if self.pre_tokenize:
            token_ids = self.samples[idx]
            if self.func_to_get_labels is not None:
                labels = self.labels[idx]
        else:
            midi = Score(self.files_paths[idx])
            token_ids = self.tokenizer.midi_to_tokens(midi)
            # If one_token_stream, we only take the first track/sequence
            if self.tokenizer.one_token_stream:
                token_ids = token_ids[0]
            if self.func_to_get_labels is not None:
                labels = self.func_to_get_labels(midi, self.files_paths[idx])

        item = {self.sample_key_name: token_ids}
        if labels is not None:
            item[self.labels_key_name] = labels

        return item

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
        return f"{len(self.files_paths)} MIDI files."


class DatasetJsonIO(_DatasetABC):
    r"""
    Basic ``Dataset`` loading Json files of tokenized MIDIs on the fly.

    When indexing it (``dataset[idx]``), this class will load the ``files_paths[idx]``
    json file and return the token ids, that can be used to train generative models.
    **This class is only compatible with tokens saved as a single stream of tokens
    (** ``tokenizer.one_token_stream`` **).** If you plan to use it with token files
    containing multiple token streams, you should first it with
    ``miditok.pytorch_data.split_dataset_to_subsequences()``.

    It allows to reduce the sequence length up to a `max_seq_len` limit, but will not
    split the sequences into subsequences. If your dataset contains sequences with
    lengths largely varying, you might want to first split it into subsequences with
    the ``miditok.pytorch_data.split_dataset_to_subsequences()`` method before loading
    it to avoid losing data.

    This ``Dataset`` class is well suited if you are using a large dataset, or have
    access to limited RAM resources.

    :param files_paths: list of paths to files to load.
    :param max_seq_len: maximum sequence length (in num of tokens). (default: ``None``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        max_seq_len: int | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.files_paths = files_paths
        super().__init__()

    def __getitem__(self, idx: int) -> Mapping[str, LongTensor]:
        """
        Load the tokens from the ``idx`` json file.

        :param idx: index of the file to load.
        :return: the tokens as a dictionary mapping to the token ids as a tensor.
        """
        with self.files_paths[idx].open() as json_file:
            token_ids = json.load(json_file)["ids"]
        if self.max_seq_len is not None and len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        return {"input_ids": LongTensor(token_ids)}

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return len(self.files_paths)
