"""Byte Pair Encoding (BPE) class wrapper, to use with MIDITokenizer child object.
This does not work with "multi-embedding" representations like CP Word or Octuple.
"""

from typing import List, Dict, Union, Type, Tuple
from pathlib import Path, PurePath
from random import choices
from copy import deepcopy

from miditoolkit import MidiFile
from tqdm import tqdm

from .midi_tokenizer_base import MIDITokenizer, Event, Vocabulary


def bpe(tokenizer: Type[MIDITokenizer], *args, **kwargs):

    class BPE(tokenizer):
        r"""A wrapper for any tokenizer object, which allows to use Byte Pair Encoding (BPE).
        """
        def __init__(self):
            self.has_bpe = False
            super().__init__(*args, **kwargs)
            self.bpe_successions = {}
            if self.has_bpe:  # loaded from config file
                self.add_bpe_to_tokens_type_graph()
                self.set_bpe_tokens_successions()
                self.vocab.update_token_types_indexes()

        def bpe(self, tokens_path: Union[Path, PurePath, str], vocab_size: int, out_dir: Union[Path, PurePath, str],
                files_lim: int = None, save_converted_samples: bool = False, print_seq_len_variation: bool = True):
            r"""Byte Pair Encoding (BPE) method to build the vocabulary.
            This method will build (modify) the vocabulary by analyzing a tokenized dataset to find
            the most recurrent token successions.
            Note that this implementation is in pure Python and will be slow if you use a large amount of
            tokens files. You might use the files_lim argument.

            :param tokens_path: path to token files to learn the BPE combinations from
            :param vocab_size: the new vocabulary size
            :param out_dir: directory to save the tokenizer's parameters and vocabulary after BPE learning is finished
            :param files_lim: limit of token files to use (default: None)
            :param save_converted_samples: will save in out_path the samples that have been used
                    to create the BPE vocab. Files will keep the same name and relative path (default: True)
            :param print_seq_len_variation: prints the mean sequence length before and after BPE,
                    and the variation in %. (default: True)
            """
            assert vocab_size > len(self.vocab), f'vocab_size ({vocab_size}) need to be higher than the size' \
                                                 f'of the current vocabulary ({len(self.vocab)})'
            if isinstance(out_dir, str):
                out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            files_paths = list(Path(tokens_path).glob('**/*.json'))
            files_paths_bpe = choices(files_paths, k=files_lim) if \
                (files_lim is not None and files_lim < len(files_paths)) else files_paths
            samples = []
            samples_paths = []
            original_lengths = []

            # Loads tokens / samples to analyze
            for file_path in tqdm(files_paths_bpe, desc='Loading token files'):
                file = self.load_tokens(file_path)
                samples.append(file)
                samples_paths.append(file_path.relative_to(tokens_path))
                original_lengths += [len(track) for track in file['tokens']]

            # Byte Pair Encoding
            pbar = tqdm(total=vocab_size - len(self.vocab), desc='Learning byte pair encoding')
            while len(self.vocab) < vocab_size:
                pbar.update(1)
                occurrences = {}  # count occurrences of successive tokens
                for sample in samples:
                    for track in sample['tokens']:
                        for i in range(len(track) - 1):
                            try:
                                occurrences[tuple(track[i: i + 2])] += 1
                            except KeyError:
                                occurrences[tuple(track[i: i + 2])] = 1

                to_replace = max(occurrences, key=occurrences.get)  # most recurrent succession of two tokens
                to_replace_bis = []  # store non-BPE tokens to be registered in vocab
                for token in to_replace:
                    if self.vocab.token_to_event[token].split('_')[0] == 'BPE':
                        to_replace_bis += map(int,
                                              self.vocab.token_to_event[token].split('_')[1].split('.')[1].split('-'))
                    else:
                        to_replace_bis.append(token)
                to_replace_str = '-'.join(map(str, to_replace)) + '.' + '-'.join(map(str, to_replace_bis))
                self.vocab.add_event(Event(type_='BPE', time=0, value=to_replace_str, desc=''))
                for sample in samples:  # replace in existing samples
                    for track in sample['tokens']:
                        i = 0
                        while i < len(track) - 1:
                            if tuple(track[i: i + 2]) == to_replace:
                                track[i] = self.vocab[f'BPE_{to_replace_str}']
                                del track[i + 1]
                            i += 1

            # Saves dictionary and prints the difference in sequence length
            pbar.close()
            self.has_bpe = True
            self.set_bpe_tokens_successions()
            self.add_bpe_to_tokens_type_graph()
            self.vocab.update_token_types_indexes()
            new_lengths = []
            for sample, path in zip(samples, samples_paths):
                if save_converted_samples:
                    self.save_tokens(sample['tokens'], PurePath(out_dir, path).with_suffix(".json"), sample['programs'])
                new_lengths += [len(track) for track in sample['tokens']]
            if print_seq_len_variation:
                original_mean = sum(original_lengths) / len(original_lengths) if len(original_lengths) > 0. else 0.
                new_mean = sum(new_lengths) / len(new_lengths) if len(new_lengths) > 0. else 0.
                print(f'Mean of original lengths: {original_mean}\nMean length after BPE: {new_mean}')
                print(f'Variation from original: {(new_mean - original_mean) / original_mean * 100:.2f} %')
            self.save_params(out_dir / 'config.txt')  # Saves the parameters with which the MIDIs are converted

        def set_bpe_tokens_successions(self):
            """Creates the bpe_successions attributes, as a dictionary of the form bpe_token: (tok1, tok2, tok3...)
            """
            self.bpe_successions = {tok: list(map(int, self.vocab.token_to_event[tok].split('_')[1].split('.')[0].
                                                  split('-'))) for tok in self.vocab.tokens_of_type('BPE')}

        def apply_bpe(self, tokens: List[int]) -> List[int]:
            r"""Converts a sequence of tokens into tokens with BPE.

            :param tokens: tokens to convert.
            :return:
            """
            if not self.has_bpe:
                return tokens

            previous_len = len(tokens) + 1  # + 1 to fool when entering the loop the first time
            while previous_len != len(tokens):  # if this is True, it means no more BPE combinations is possible
                previous_len = len(tokens)  # length of the token sequence before applying BPE
                for tok, token_succession in self.bpe_successions.items():  # loops over BPE tokens from the vocabulary
                    occurrences = self.__subfind(tokens, token_succession)
                    for idx in reversed(occurrences):
                        tokens[idx] = tok
                        for _ in range(len(token_succession) - 1):
                            del tokens[idx + 1]
            return tokens

        @staticmethod
        def __subfind(in_list: List[int], pattern: List[int]) -> List[int]:
            """Finds the locations of a pattern within a list.
            Adapted from: https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
            Related: https://www.reddit.com/r/learnpython/comments/2xqlwj/using_npwhere_to_find_subarrays/
            After testing, the numba jit version does not seem to be much faster.
            The conversion of python lists to numba.typed.List() seems to also take time.

            :param in_list: input list to analyze
            :param pattern: pattern to detect
            :return: indices of in_list where the pattern has been found
            """
            matches = []
            for i in range(len(in_list)):
                if in_list[i] == pattern[0] and in_list[i:i + len(pattern)] == pattern:
                    matches.append(i)
            return matches

        def apply_bpe_to_dataset(self, dataset_path: Union[Path, PurePath, str], out_path: Union[Path, PurePath, str]):
            r"""Apply BPE to an already tokenized dataset (with no BPE).

            :param dataset_path: path to token files to load
            :param out_path: output directory to save
            """
            if not self.has_bpe:
                return

            files_paths = list(Path(dataset_path).glob('**/*.json'))
            for path in tqdm(files_paths, desc='Applying BPE to dataset'):
                sample = self.load_tokens(path)
                sample_bpe = [self.apply_bpe(track) for track in sample['tokens']]
                self.save_tokens(sample_bpe, Path(out_path) / path.relative_to(dataset_path), sample['programs'])

        def midi_to_tokens(self, midi: MidiFile, *args_, **kwargs_) -> List[List[int]]:
            r"""First convert the MIDI into "regular" tokens, then apply BPE.

            :param midi: MIDI object to convert.
            :return: the token representation, i.e. tracks converted into sequences of tokens
            """
            tokens = super().midi_to_tokens(midi, *args_, **kwargs_)
            return [self.apply_bpe(track) for track in tokens] if self.has_bpe else tokens

        def tokens_to_events(self, tokens: List[int], return_decomposed_tokens: bool = False, **kwargss) \
                -> Union[List[Event], Tuple[List[Event], List[int]]]:
            r"""First decomposes BPE tokens, then converts them in their respective event objects.

            :param tokens: sequence of tokens to convert
            :param return_decomposed_tokens: set True if you want the decomposed tokens to be returned (default: False)
            :return: the sequence of corresponding events
            """
            # Decompose BPE tokens first
            decomposed_tokens = self.decompose_bpe(deepcopy(tokens))
            events = super().tokens_to_events(decomposed_tokens, **kwargss)
            return (events, decomposed_tokens) if return_decomposed_tokens else events

        def decompose_bpe(self, tokens: List[int]) -> List[int]:
            r"""Decomposes a sequence of tokens containing BP encoded tokens into "prime" tokens.
            It is an inplace operation.

            :param tokens: token sequence to decompose
            :return: decomposed token sequence
            """
            i = 0
            while i < len(tokens):
                token_type, token_val = self.vocab[tokens[i]].split('_')
                if token_type == 'BPE':
                    del tokens[i]
                    for j, to_insert in enumerate(map(int, token_val.split('.')[1].split('-'))):
                        tokens.insert(i + j, to_insert)
                i += 1
            return tokens

        def token_types_errors(self, tokens: List[int], consider_pad: bool = False) -> float:
            r"""Checks if a sequence of tokens is constituted of good token types
            successions and returns the error ratio (lower is better).
            The Pitch and Position values are also analyzed:
                - a position token cannot have a value <= to the current position (it would go back in time)
                - a pitch token should not be present if the same pitch is already played at the current position
            :param tokens: sequence of tokens to check
            :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
            :return: the error ratio (lower is better)
            """
            # Decompose BPE tokens first
            tokens = self.decompose_bpe(tokens)
            return super().token_types_errors(tokens, consider_pad)

        def add_bpe_to_tokens_type_graph(self):
            r"""Adds BPE to the tokens_types_graph.
            You must manually call this method after loading a BPE tokenizer from params (config file) if
            you intend to use tokens_types_graph.
            """
            for val in self.tokens_types_graph.values():
                val.append('BPE')
            self.tokens_types_graph['BPE'] = list(self.tokens_types_graph.keys())

        def save_params(self, out_dir: Union[str, Path, PurePath], additional_attributes: Dict = None):
            r"""Saves the config / base parameters of the tokenizer in a file.
            Useful to keep track of how a dataset has been tokenized / encoded
            It will also save the name of the class used, i.e. the encoding strategy.
            NOTE: the vocabulary (token_to_event) will be saved with the 'vocab' key, that will be decoded
                back by the load_params method.
            NOTE 2: as json cant save tuples as keys, the beat ranges are saved as strings
                with the form startingBeat_endingBeat (underscore separating these two values)

            :param out_dir: output directory to save the file
            :param additional_attributes: any additional information to store in the config file. (default: None)
            """
            if additional_attributes is None:
                additional_attributes = {}
            additional_attributes['vocab'] = self.vocab.token_to_event
            super().save_params(out_dir, additional_attributes)

        def load_params(self, params: Union[str, Path, PurePath]):
            r"""Load parameters and set the encoder attributes

            :param params: can be a path to the parameter (json encoded) file
            """
            super().load_params(params)
            token_to_event = deepcopy(self.vocab)  # from the saved config file
            self.vocab = Vocabulary()
            self.vocab._token_to_event = {int(token): event for token, event in token_to_event.items()}
            self.vocab._event_to_token = {event: int(token) for token, event in token_to_event.items()}
            self.vocab.update_token_types_indexes()
            self.has_bpe = len(self.vocab.tokens_of_type('BPE')) > 0

    # tokenizer.__class__.__bases__ += (BPE,)
    return BPE()
