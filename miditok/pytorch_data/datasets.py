"""Dataset classes to be used with PyTorch when training a model."""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any
from warnings import warn

from symusic import Score
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE
from miditok.utils import split_midi_per_tracks

from .split_midi_utils import (
    get_average_num_tokens_per_note,
    split_midi_per_note_density,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from miditok import MIDITokenizer, TokSequence


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
        token_ids: list[int],
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> list[int]:
        # Reduce sequence length
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        # Adds BOS and EOS tokens
        if bos_token_id:
            token_ids.insert(0, bos_token_id)
        if eos_token_id:
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
    A``Dataset`` loading and tokenizing MIDIs during training.

    This class can be used for several strategies, with two major independent options:
    MIDI splitting and pre-tokenizing.

    # TODO document MIDI split and global workflow
    **MIDI splitting** will allow to split each MIDI from the original dataset chunks
    of lengths calculated in function of the note densities of its bars in order to
    reduce the padding of the batches. MIDI splitting can be controlled with the
    ``split_midis``, ``average_num_tokens_per_note``, ``save_dir`` and
    ``split_kwargs`` arguments.
    # TODO done once with save_dir
    # TODO if no one_token_stream --> Tracks will be separated

    **Pre-tokenizing** signifies that the ``DatasetMIDI`` will tokenize the MIDI
    dataset at its initialization, and store the token ids in RAM.

    Additionally, you can use the ``func_to_get_labels`` argument to provide a method
    allowing to use labels (one label per file).

    :param files_paths: paths to MIDI files to load.
    :param tokenizer: tokenizer.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    :param split_midis:
    :param split_kwargs: keyword arguments to pass to the
        :py:func:`miditok.pytorch_data.save_pretrained` method. Note that if the
        ``average_num_tokens_per_note`` is not given, it will automatically be
        calculated from the first 200 MIDI files with the
        :py:func:`miditok.pytorch_data.get_average_num_tokens_per_note` method.
    :param save_dir:
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
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        split_midis: bool = True,
        split_kwargs: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        pre_tokenize: bool = False,
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
        self.func_to_get_labels = func_to_get_labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        self.samples, self.labels = ([], []) if func_to_get_labels else (None, None)
        self._effective_max_seq_len = max_seq_len - sum(
            [1 for tok in [bos_token_id, eos_token_id] if tok is not None]
        )

        # If MIDI splitting, check the save_dir doesn't already contain them
        midi_split_hidden_file_path = None
        if split_midis:
            if save_dir is None:
                save_dir = Path(mkdtemp(prefix="miditok-dataset"))
                warn(
                    f"The MIDI splits will be saved in a temporary directory: "
                    f"{save_dir}",
                    stacklevel=2,
                )
            midi_split_hidden_file_path = save_dir / f".{hash(tuple(self.files_paths))}"
            if midi_split_hidden_file_path.is_file():
                self.files_paths = list(save_dir.glob("**/*.mid"))
                # Disable split, no need anymore
                split_midis = False
            if not split_kwargs:
                split_kwargs = {}
            if not split_kwargs.get("average_num_tokens_per_note"):
                split_kwargs[
                    "average_num_tokens_per_note"
                ] = get_average_num_tokens_per_note(
                    tokenizer, self.files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
                )

        # Load the MIDIs here to split and/or pre-tokenize them
        if split_midis or pre_tokenize:
            # Find the highest common subdir
            root_dir = None
            if split_midis and save_dir:
                all_parts = [path.parent.parts for path in self.files_paths]
                max_depth = max(len(parts) for parts in all_parts)
                root_parts = []
                for depth in range(max_depth):
                    if len({parts[depth] for parts in all_parts}) > 1:
                        break
                    root_parts.append(all_parts[0][depth])
                root_dir = Path(*root_parts)

            # pbar desc
            if split_midis and pre_tokenize:
                pbar_desc = f"Splitting MIDIs ({save_dir}) and pre-tokenizing"
            elif split_midis:
                pbar_desc = f"Splitting MIDIs ({save_dir})"
            else:
                pbar_desc = "Pre-tokenizing"

            new_files_paths = []
            for file_path in tqdm(
                self.files_paths,
                desc=pbar_desc,
                miniters=int(len(self.files_paths) / 20),
                maxinterval=480,
            ):
                midis = [Score(file_path)]

                # Split MIDI if requested
                # TODO separate from the DatasetMIDI class?
                if split_midis:
                    # Separate track first if needed, but only if we do not
                    # pre-tokenize as it would otherwise increase the overall duration
                    # by pre-processing MIDIs multiple times pointlessly.
                    tracks_separated = False
                    if (
                        not tokenizer.one_token_stream
                        and not pre_tokenize
                        and len(midis[0].tracks) > 1
                    ):
                        midis = split_midi_per_tracks(midis[0])
                        tracks_separated = True

                    # Split per note density
                    midis_splits = []
                    for ti, midi_to_split in enumerate(midis):
                        midi_splits = split_midi_per_note_density(
                            midi_to_split,
                            self._effective_max_seq_len,
                            **split_kwargs,
                        )

                        # Save them
                        for _i, midi_to_save in enumerate(midi_splits):
                            # Skip it if there are no notes, this can happen with
                            # portions of tracks with no notes but tempo/signature
                            # changes happening later
                            if (
                                len(midi_to_save.tracks) == 0
                                or midi_to_save.note_num() == 0
                            ):
                                continue
                            if tracks_separated:
                                file_name = f"{file_path.stem}_t{ti}_{_i}.mid"
                            else:
                                file_name = f"{file_path.stem}_{_i}.mid"
                            # use with_stem when dropping support for python 3.8
                            saving_path = (
                                save_dir
                                / file_path.relative_to(root_dir).parent
                                / file_name
                            )
                            saving_path.parent.mkdir(parents=True, exist_ok=True)
                            midi_to_save.dump_midi(saving_path)
                            new_files_paths.append(saving_path)
                            midis_splits.append(midi_to_save)
                    # Reassign original MIDI(s) with splits
                    midis = midis_splits

                # Pre-tokenize the MIDI(s) and store the tokens in memory
                if pre_tokenize:
                    for midi in midis:
                        tokseq = self._tokenize_midi(midi)
                        if tokenizer.one_token_stream:
                            tokseq = [tokseq]
                        for seq in tokseq:
                            self.samples.append(LongTensor(seq.ids))
                            if func_to_get_labels:
                                label = func_to_get_labels(midi, seq, file_path)
                                if not isinstance(label, LongTensor):
                                    label = LongTensor(label)
                                self.labels.append(label)

            # Save file in save_dir to indicate MIDI split has been performed
            if split_midis:
                self.files_paths = new_files_paths
                with midi_split_hidden_file_path.open("w") as f:
                    f.write(f"{len(self.files_paths)} files after MIDI splits")

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Return the `idx` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the `idx`th MIDI file on the fly.

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
            midi = Score(self.files_paths[idx])
            tokseq = self._tokenize_midi(midi)
            # If not one_token_stream, we only take the first track/sequence
            token_ids = tokseq.ids if self.tokenizer.one_token_stream else tokseq[0].ids
            if self.func_to_get_labels is not None:
                # tokseq can be given as a list of TokSequence to get the labels
                labels = self.func_to_get_labels(midi, tokseq, self.files_paths[idx])
                if not isinstance(labels, LongTensor):
                    labels = LongTensor(labels)

        item = {self.sample_key_name: LongTensor(token_ids)}
        if labels is not None:
            item[self.labels_key_name] = labels

        return item

    def _tokenize_midi(self, midi: Score) -> TokSequence | list[TokSequence]:
        # Tokenize
        tokseq = self.tokenizer.midi_to_tokens(midi)

        # If tokenizing on the fly a multi-stream tokenizer, only keeps the first track
        if not self.pre_tokenize and not self.tokenizer.one_token_stream:
            tokseq = [tokseq[0]]

        # Preprocessing token ids
        # TODO only BOS when real beginning of a MIDI (first chunk of split)
        # TODO only EOS when real end of a MIDI (last chunk of split)
        if self.tokenizer.one_token_stream:
            tokseq.ids = self._preprocess_token_ids(
                tokseq.ids,
                self._effective_max_seq_len,
                self.bos_token_id,
                self.eos_token_id,
            )
        else:
            for seq in tokseq:
                seq.ids = self._preprocess_token_ids(
                    seq.ids,
                    self._effective_max_seq_len,
                    self.bos_token_id,
                    self.eos_token_id,
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
        Load the tokens from the ``idx`` json file.

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
