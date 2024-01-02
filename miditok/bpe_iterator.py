"""
Common classes.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from symusic import Score

from .constants import MIDI_FILES_EXTENSIONS, MIDI_LOADING_EXCEPTION

if TYPE_CHECKING:
    from .midi_tokenizer import MIDITokenizer


class BPEIterator:
    r"""Iterator class that loads MIDI files and tokenize them on the fly, to be used
    with the Hugging Face tokenizers library to build a vocabulary with BPE.

    :param tokenizer: tokenizer to use for training.
    :param files_paths: sequence of paths of files to load for training.
    """

    def __init__(self, tokenizer: MIDITokenizer, files_paths: Sequence[Path]) -> None:
        self.tokenizer = tokenizer
        self.files_paths = files_paths
        self.__iter_count = 0

    def load_file(self, path: Path) -> list[str]:
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
        return len(self.files_paths)

    def __getitem__(self, idx: int) -> list[str]:
        return self.load_file(self.files_paths[idx])

    def __iter__(self) -> BPEIterator:
        return self

    def __next__(self) -> list[str]:
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration
        else:
            self.__iter_count += 1
            return self[self.__iter_count - 1]

    def __str__(self) -> str:
        return f"{self.tokenizer} - {len(self)} files"
