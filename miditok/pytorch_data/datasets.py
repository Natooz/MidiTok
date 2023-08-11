"""
PyTorch `Dataset` objects, to be used with PyTorch `DataLoaders` to load and send data during training.
"""
from pathlib import Path
from typing import List, Union, Sequence, Any, Mapping
from abc import ABC
from copy import deepcopy
import json

from torch import LongTensor, randint
from torch.utils.data import Dataset
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from tqdm import tqdm

from ..constants import MIDI_FILES_EXTENSIONS


def split_seq_in_subsequences(
    seq: Sequence[any], min_seq_len: int, max_seq_len: int
) -> List[Sequence[Any]]:
    r"""Split a sequence of tokens into subsequences for which `min_seq_len <= len(sub_seq) <= max_seq_len`.

    :param seq: sequence to split.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :return: list of subsequences.
    """
    sub_seq = []
    i = 0
    while i < len(seq):
        if i >= len(seq) - min_seq_len:
            break  # last sample is too short
        sub_seq.append(LongTensor(seq[i : i + max_seq_len]))
        i += len(sub_seq[-1])  # could be replaced with max_seq_len

    return sub_seq


def split_dataset_to_subsequences(
    files_paths: Sequence[Union[Path, str]],
    out_dir: Union[Path, str],
    min_seq_len: int,
    max_seq_len: int,
    one_token_stream: bool = True,
):
    """

    :param files_paths: list of files of tokens to split.
    :param out_dir: output directory to save the subsequences.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :param one_token_stream: give False if the token files contains multiple tracks, i.e. the first dimension of
        the value of the "ids" entry corresponds to several tracks. Otherwise, leave False. (default: True)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files_paths:
        with open(file_path) as json_file:
            tokens = json.load(json_file)

        # Split sequence(s)
        if one_token_stream:
            subseqs = split_seq_in_subsequences(tokens["ids"], min_seq_len, max_seq_len)
        else:
            subseqs = []
            for track_seq in tokens["ids"]:
                subseqs += split_seq_in_subsequences(
                    track_seq, min_seq_len, max_seq_len
                )

        # Save subsequences
        for i, subseq in enumerate(subseqs):
            path = out_dir / f"{file_path.name}_{i}.json"
            with open(path, "w") as outfile:
                new_tok = deepcopy(tokens)
                new_tok["ids"] = subseq
                json.dump(tokens, outfile)


class _DatasetABC(Dataset, ABC):
    r"""Abstract Dataset class, holding samples (and optionally labels) and implementing the basic magic methods.

    :param samples: sequence of input samples. It can directly be data, or a paths to files to be loaded.
    :param labels: sequence of labels associated with the samples. (default: None)
    :param sample_key_name: name of the dictionary key containing the sample data when iterating the dataset.
            (default: "input_ids")
    :param labels_key_name: name of the dictionary key containing the labels data when iterating the dataset.
            (default: "labels")
    """

    def __init__(
        self,
        samples: Sequence[Any] = None,
        labels: Sequence[Any] = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ):
        if samples is not None and labels is not None:
            assert len(samples) == len(
                labels
            ), "The number of samples must be the same as the number of labels"
        self.samples = samples if samples is not None else []
        self.labels = labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name

    def reduce_nb_samples(self, nb_samples: int):
        r"""Reduce the size of the dataset, by keeping `nb_samples` samples.

        :param nb_samples: number of samples to keep. They will be randomly picked.
        """
        idx = randint(0, len(self), (nb_samples,))
        self.samples = [self.samples[id_] for id_ in idx.tolist()]
        if self.labels is not None:
            self.labels = [self.labels[id_] for id_ in idx.tolist()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Mapping[str, Any]:
        item = {self.sample_key_name: self.samples[idx]}
        if self.labels is not None:
            item[self.labels_key_name] = self.labels[idx]

        return item

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "No data loaded" if len(self) == 0 else f"{len(self.samples)} samples"


class DatasetTok(_DatasetABC):
    r"""Basic `Dataset` loading Json files of tokenized MIDIs, or MIDI files to tokenize, and
    store the token ids in RAM. It outputs token sequences that can be used to train generative models.

    The tokens sequences being loaded will then be split into subsequences, of length
    comprise between `min_seq_len` and `max_seq_len`.
    For example, with `min_seq_len = 50` and `max_seq_len = 100`:
    * a sequence of 650 tokens will be split into 6 subsequences of 100 tokens plus one subsequence of 50 tokens;
    * a sequence of 620 tokens will be split into 6 subsequences of 100 tokens, the last 20 tokens will be discarded;
    * a sequence of 670 tokens will be split into 6 subsequences of 100 tokens plus one subsequence of 50 tokens,
        and the last 20 tokens will be discarded.

    This `Dataset` class is well suited if you have enough RAM to store all the data, as it does not require you to
    prior split the dataset into subsequences of the length you desire.
    Note that if you directly load MIDI files, the loading can take some time as they will need to be tokenized.
    You might want to tokenize them before once with the `tokenizer.tokenize_midi_dataset()` method.

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    :param one_token_stream: give False if the token files contains multiple tracks, i.e. the first dimension of
        the value of the "ids" entry corresponds to several tracks. Otherwise, leave False. (default: True)
    :param sample_key_name: name of the dictionary key containing the sample data when iterating the dataset.
            (default: "input_ids")
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        min_seq_len: int,
        max_seq_len: int,
        tokenizer: MIDITokenizer = None,
        one_token_stream: bool = True,
        sample_key_name: str = "input_ids",
    ):
        samples = []
        if tokenizer is not None:
            one_token_stream = tokenizer.one_token_stream

        for file_path in tqdm(
            files_paths,
            desc=f"Loading data: {files_paths[0].parent}",
            miniters=int(len(files_paths) / 20),
            maxinterval=480,
        ):
            if file_path.suffix in MIDI_FILES_EXTENSIONS:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)
                if one_token_stream:
                    tokens = tokens.ids
                else:
                    tokens = [seq.ids for seq in tokens]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)["ids"]
            if one_token_stream:
                tokens = [tokens]

            # Cut tokens in samples of appropriate length
            for seq in tokens:
                subseqs = split_seq_in_subsequences(seq, min_seq_len, max_seq_len)
                samples += subseqs

        super().__init__(samples, sample_key_name=sample_key_name)


class DatasetJsonIO(_DatasetABC):
    r"""Basic `Dataset` loading Json files of tokenized MIDIs on the fly.
    When executing `dataset[idx]`, this class will load the `files_paths[idx]` json file and return the token ids,
    that can be used to train generative models.
    **This class is only compatible with tokens saved as a single stream of tokens
    (** `tokenizer.one_token_stream` **).** If you plan to use it with token files containing multiple token streams,
    you should first it with `miditok.pytorch_data.split_dataset_to_subsequences()`.

    It allows to reduce the sequence length up to a `max_seq_len` limit, but will not split the sequences into
    subsequences. If your dataset contains sequences with lengths largely varying, you might want to first split it
    into subsequences with the `miditok.pytorch_data.split_dataset_to_subsequences()` method before loading it to avoid
    losing data.

    This `Dataset` class is well suited if you are using a large dataset, or have access to limited RAM resources.

    :param files_paths: list of paths to files to load.
    :param max_seq_len: maximum sequence length (in nb of tokens). (default: None)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        max_seq_len: int = None,
    ):
        self.max_seq_len = max_seq_len
        super().__init__(files_paths)

    def __getitem__(self, idx) -> Mapping[str, LongTensor]:
        with open(self.samples[idx]) as json_file:
            token_ids = json.load(json_file)["ids"]
        if self.max_seq_len is not None and len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        item = {"input_ids": LongTensor(token_ids)}

        return item
