"""Dataset classes to be used with PyTorch when training a model."""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

import numpy as np
from symusic import Score
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE
from miditok.utils import (
    get_bars_ticks,
    get_beats_ticks,
    get_num_notes_per_bar,
    split_midi_per_ticks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from miditok import MIDITokenizer


def get_distribution_num_tokens_per_beat(
    files_paths: Sequence[Path], tokenizer: MIDITokenizer
) -> list[float]:
    """
    Return the distributions of number of tokens per beat for a list of files.

    :param files_paths: paths to the files to load.
    :param tokenizer: tokenizer.
    :return: the distribution of number of tokens per beat for each file, and/or
        each track if ``tokenizer.one_token_stream`` is ``False``.
    """
    tpb_dist = []
    for files_path in tqdm(
        files_paths, desc="Calculating the tokens/beat distribution"
    ):
        # Load MIDI and tokenize it
        midi = Score(files_path)
        ticks_beats = get_beats_ticks(midi)
        num_beats_global = len(ticks_beats)
        tokens = tokenizer(midi)

        if tokenizer.one_token_stream:
            tpb_dist.append(len(tokens) / num_beats_global)
        else:
            ticks_beats = np.array(get_beats_ticks(midi))
            for track, seq in zip(midi.tracks, tokens):
                # track.start is always 0, so we use the first note's time
                beat_start = np.where((ticks_beats - track.notes[0].time) <= 0)[0][-1]
                beat_end = np.where((ticks_beats - track.end()) >= 0)[0]
                beat_end = num_beats_global if len(beat_end) == 0 else beat_end[0]
                tpb_dist.append(len(seq) / (beat_end - beat_start))

    return tpb_dist


def get_num_beats_for_token_seq_len(
    files_paths: Sequence[Path],
    tokenizer: MIDITokenizer,
    sequence_length: int,
    ratio_of_full_sequence_length: float,
) -> float:
    """
    Return the number of beats covering *x*% of the sequences of *y* tokens.

    This method calls
    :py:func:`miditok.pytorch_data.get_num_tokens_per_beat_distribution` and returns
    the number of beats covering ``ratio_of_data_to_keep``% of the sequences of
    ``sequence_length`` tokens.
    This method is useful to calculate the appropriate chunk length in beats to use
    with the :py:func:`miditok.utils.split_midi` method (also called by
    :class:`miditok.pytorch_data.DatasetTok`).

    :param files_paths: paths to the files to load.
    :param tokenizer: tokenizer.
    :param sequence_length: number of tokens in the sequence.
    :param ratio_of_full_sequence_length: ratio of sequences that must contain at most
        `sequence_length`` tokens.
    :return: number of beats covering *x*% of the sequences of *y* tokens.
    """
    tpb_dist = get_distribution_num_tokens_per_beat(files_paths, tokenizer)
    bpt_dist = np.reciprocal(np.array(tpb_dist)) * sequence_length
    bpt_dist.sort()
    return np.percentile(bpt_dist, ratio_of_full_sequence_length * 100)


def split_midi_per_note_density(
    midi: Score,
    max_seq_len: int,
    average_num_tokens_per_note: float,
    one_token_stream: bool,
) -> list[Score]:
    """
    Split a MIDI into chunks depending on their note density.

    Using note densities aims to reduce the amount of padding when splitting the MIDIs.

    :param midi: MIDI to split.
    :param max_seq_len: maximum number of tokens per sequence.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer.
    :param one_token_stream: whether the tokenizer is ``one_token_stream``.
    :return: the list of split MIDIs.
    """
    bar_ticks = get_bars_ticks(midi)
    num_notes_per_bar = get_num_notes_per_bar(
        midi,
        tracks_indep=~one_token_stream,
    )
    num_tokens_per_bar = [
        npb * average_num_tokens_per_note for npb in num_notes_per_bar
    ]
    ticks_split = []
    num_tokens = 0
    for bi, tpb in enumerate(num_tokens_per_bar):  # TODO use bi
        next_num_tokens = num_tokens + tpb
        if next_num_tokens > max_seq_len:
            ticks_split = bar_ticks[bi]  # TODO bi good??
        else:
            num_tokens = next_num_tokens

    return split_midi_per_ticks(midi, ticks_split)


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


class DatasetMIDI(_DatasetABC):
    r"""
    A``Dataset`` loading and tokenizing MIDIs during training.

    This class can be used for several strategies, with two major independent options:
    MIDI splitting and pre-tokenizing.

    **Pre-tokenizing** signifies that the ``DatasetMIDI`` will tokenize the MIDI
    dataset at its initialization, and store the token ids in RAM.

    Additionally, you can use the ``func_to_get_labels`` argument to provide a method
    allowing to use labels (one label per file).

    **Note:** if your tokenizer tokenizes tracks independently (``one_token_stream``
    property), then if you don't pre-tokenize the dataset, only the tokens of the first
    track will be used.
    # TODO splitting: minimize padding or maximize amount of data --> sort samples?

    :param files_paths: paths to MIDI files to load.
    :param tokenizer: tokenizer.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param save_dir:
    :param save_dir_tmp:
    :param pre_tokenize:
    :param func_to_get_labels: a function to retrieve the label of a file. The method
        must take two positional arguments: the first is either the
        :class:`miditok.TokSequence` returned when tokenizing a MIDI, the second is the
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
        tokenizer: MIDITokenizer,
        max_seq_len: int,
        split_midis: bool = True,
        average_num_tokens_per_note: float | None = None,
        save_dir: Path | None = None,
        save_dir_tmp: bool = False,
        pre_tokenize: bool = False,
        func_to_get_labels: Callable[[Score | Sequence, Path], int] | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        super().__init__()

        # If MIDI splitting, check the save_dir doesn't already contain them
        midi_split_hidden_file_path = None
        if split_midis:
            if save_dir is None:
                if save_dir_tmp:
                    save_dir = Path(mkdtemp(prefix="miditok-dataset"))
                else:
                    msg = (
                        "When splitting MIDIs, you must give a `save_dir` or use a "
                        "temporary directory (`save_dir_tmp`) to save them"
                    )
                    raise ValueError(msg)
            midi_split_hidden_file_path = save_dir / f".{hash(files_paths)}"
            if midi_split_hidden_file_path.is_file():
                files_paths = list(save_dir.glob("**/*.mid"))
                # Disable split, no need anymore
                split_midis = False
            elif average_num_tokens_per_note is None:
                get_average_num_tokens_per_note(
                    files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
                )

        # Set class attributes
        self.files_paths = files_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pre_tokenize = pre_tokenize
        self.func_to_get_labels = func_to_get_labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        self.samples, self.labels = ([], []) if func_to_get_labels else (None, None)

        # TODO rolling note density (per bar?) --> estimate num_tokens per bar -->
        # notes/bar --> tokens/notes --> estimate tokens/bar

        # Load the MIDIs here to split and/or pre-tokenize them
        if split_midis or pre_tokenize:
            # Find the highest common subdir
            root_dir = None
            if split_midis and save_dir:
                all_parts = [path.parent.parts for path in files_paths]
                max_depth = max(len(parts) for parts in all_parts)
                root_parts = []
                for depth in range(max_depth):
                    if len({parts[depth] for parts in all_parts}) > 1:
                        break
                    root_parts.append(all_parts[0][depth])
                root_dir = Path(*root_parts)

            for file_path in tqdm(
                files_paths,
                desc=f"Preprocessing files: {files_paths[0].parent}",
                miniters=int(len(files_paths) / 20),
                maxinterval=480,
            ):
                # Split MIDI if requested
                if split_midis:
                    midis = split_midi_per_note_density(
                        Score(file_path),
                        max_seq_len,
                        average_num_tokens_per_note,
                        tokenizer.one_token_stream,
                    )
                    # Save them if needed
                    if save_dir:
                        for _i, midi in enumerate(midis):
                            saving_path = save_dir / file_path.relative_to(
                                root_dir
                            ).with_stem(f"{file_path.stem}_{_i}")
                            midi.dump(saving_path)
                            self.files_paths.append(saving_path)
                else:
                    midis = [Score(file_path)]

                # Pre-tokenize the MIDI(s) and store the tokens in memory
                if pre_tokenize:
                    for midi in midis:
                        tokseq = tokenizer.midi_to_tokens(midi)
                        if tokenizer.one_token_stream:
                            tokseq = [tokseq]
                        for seq in tokseq:
                            self.samples.append(LongTensor(seq.ids))
                            if func_to_get_labels:
                                self.labels.append(func_to_get_labels(seq, file_path))

            # Save file in save_dir to indicate MIDI split has been performed
            if split_midis:
                with midi_split_hidden_file_path.open("w") as f:
                    f.write(f"{len(self.files_paths)} files after MIDI splits")

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
