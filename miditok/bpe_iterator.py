"""Iterator to be used when training a tokenizer with the 🤗tokenizers library."""
from __future__ import annotations

from typing import TYPE_CHECKING

from symusic import Score

from .constants import MIDI_FILES_EXTENSIONS, MIDI_LOADING_EXCEPTION

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .midi_tokenizer import MIDITokenizer


class BPEIterator:
    r"""
    An iterable class to be used when training a tokenizer with BPE.

    It loads MIDI files and tokenize them on the fly, to be used with the Hugging Face
    tokenizers library to build a vocabulary with BPE.

    :param tokenizer: tokenizer to use for training.
    :param files_paths: sequence of paths of files to load for training.
    """

    def __init__(self, tokenizer: MIDITokenizer, files_paths: Sequence[Path]) -> None:
        self.tokenizer = tokenizer
        self.files_paths = files_paths
        self.__iter_count = 0

    def load_file(self, path: Path) -> list[str]:
        """
        Load a MIDI file and convert it to its byte representation.

        :param path: path to the file to load.
        :return: the byte representation of the file.
        """
        if path.suffix in MIDI_FILES_EXTENSIONS:
            try:
                midi = Score(path)
            except MIDI_LOADING_EXCEPTION:
                return []
            token_ids = self.tokenizer(midi)
            if self.tokenizer.one_token_stream:
                token_ids = token_ids.ids
            else:
                token_ids = [seq.ids for seq in token_ids]
        else:
            token_ids = self.tokenizer.load_tokens(path)["ids"]

        # list of str (bytes)
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if self.tokenizer.one_token_stream:
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

    def __iter__(self) -> BPEIterator:  # noqa:D105
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
