"""Byte Pair Encoding (BPE) class wrapper, to use with MIDITokenizer child object.
This does not work with
"""

from typing import List, Dict, Union, Any, Type
from pathlib import Path, PurePath
import json
from random import choices

from miditoolkit import MidiFile
from tqdm import tqdm

from .midi_tokenizer_base import MIDITokenizer, Event, Vocabulary


def bpe(tokenizer: Type[MIDITokenizer], *args, **kwargs):

    class BPE(tokenizer):
        r"""A wrapper for any tokenizer object, which allows to use Byte Pair Encoding (BPE).
        """
        def __init__(self):
            super().__init__(*args, **kwargs)
            self.has_bpe = False

        def bpe(self, tokens_path: Union[Path, PurePath, str], vocab_size: int, out_dir: Union[Path, PurePath, str],
                files_lim: int = None, save_converted_samples: bool = False):
            r"""Byte Pair Encoding (BPE) method.
            This method will build (modify) the vocabulary by analyzing a tokenized dataset to find
            the most recurrent token successions.
            Note that this implementation is in pure Python and will be slow if you use a large amount of
            tokens files. You might use the files_lim argument.

            :param tokens_path: path to token files to load
            :param vocab_size: the new vocabulary size
            :param out_dir: directory to save the tokenizer's parameters and vocabulary
            :param files_lim: limit of token files to use (default: None)
            :param save_converted_samples: will save in out_dir the samples that have been used
                    to create the BPE vocab. Files will keep the same name and relative path (default: True)
            """
            assert vocab_size > len(self.vocab), f'vocab_size ({vocab_size}) need to be higher than the size' \
                                                 f'of the current vocabulary ({len(self.vocab)})'
            if isinstance(out_dir, str):
                out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            files_paths = list(Path(tokens_path).glob('**/*.json'))
            files_paths_bpe = choices(files_paths, k=files_lim) if files_lim is not None else files_paths
            samples = []
            samples_paths = []
            original_lengths = []

            # Loads tokens / samples to analyze
            for file_path in tqdm(files_paths_bpe, desc='Loading token files'):
                with open(file_path) as json_file:
                    samples.append(json.load(json_file)[0][0])  # first [0] for tracks, second [0] for the first track
                    samples_paths.append(file_path.relative_to(tokens_path))
                original_lengths.append(len(samples[-1]))

            # Byte Pair Encoding
            pbar = tqdm(total=vocab_size - len(self.vocab), desc='Running byte pair encoding')
            while len(self.vocab) < vocab_size:
                pbar.update(1)
                occurrences = {}  # count occurrences of successive tokens
                for sample in samples:
                    for i in range(len(sample) - 1):
                        try:
                            occurrences[tuple(sample[i: i + 2])] += 1
                        except KeyError:
                            occurrences[tuple(sample[i: i + 2])] = 1

                to_replace = max(occurrences, key=occurrences.get)
                to_replace_bis = []  # store non-BPE tokens to be registered in vocab
                for token in to_replace:
                    if self.vocab.token_to_event[token].split('_')[0] == 'BPE':
                        to_replace_bis += map(int, self.vocab.token_to_event[token].split('_')[1].split('-'))
                    else:
                        to_replace_bis.append(token)
                to_replace_str = '-'.join(map(str, to_replace_bis))
                self.vocab.add_event(Event(type_='BPE', time=0, value=to_replace_str, desc=''))
                for sample in samples:  # replace in existing samples
                    i = 0
                    while i < len(sample) - 1:
                        if tuple(sample[i: i + 2]) == to_replace:
                            sample[i] = self.vocab[f'BPE_{to_replace_str}']
                            del sample[i + 1]
                        i += 1

            # Saves dictionary and prints the difference in sequence length
            pbar.close()
            self.has_bpe = True
            new_lengths = []
            for i, sample in enumerate(samples):
                if save_converted_samples:
                    self.save_tokens([sample], PurePath(out_dir, samples_paths[i]).with_suffix(".json"), [(0, False)])
                new_lengths.append(len(sample))
            original_mean = sum(original_lengths) / len(original_lengths) if len(original_lengths) > 0. else 0.
            new_mean = sum(new_lengths) / len(new_lengths) if len(new_lengths) > 0. else 0.
            print(f'Mean of original lengths: {original_mean}\nMean length after BPE: {new_mean}')
            print(f'Variation from original: {(new_mean - original_mean) / original_mean * 100}')
            self.save_params(out_dir)  # Saves the parameters with which the MIDIs are converted

        def convert_tokens_to_bpe(self, tokens: List[int]) -> List[int]:
            if not self.has_bpe:
                return tokens
            # while one bpe replacement in new seq
            # for bpe token in vocab, if succession of tok in seq --> replace
            while True:
                for tok in self.vocab.tokens_of_type('BPE'):
                    token_succession = 0
                    toto = 1
            return tokens

        def midi_to_tokens(self, midi: MidiFile, *args_, **kwargs_) -> List[List[int]]:
            tokens = super().midi_to_tokens(midi, *args_, **kwargs_)
            return [self.convert_tokens_to_bpe(track) for track in tokens] if self.has_bpe else tokens

        def tokens_to_events(self, tokens: List[int], **kwargss) -> List[Event]:
            r"""First decomposes BPE tokens, then converts them in their respective event objects.

            :param tokens: sequence of tokens to convert
            :return: the sequence of corresponding events
            """
            # Decompose BPE tokens first
            tokens = self.decompose_bpe(tokens)
            return super().tokens_to_events(tokens, **kwargss)

        def decompose_bpe(self, tokens: List[int]) -> List[int]:
            r"""Decomposes a sequence of tokens containing BP encoded tokens into "prime" tokens.

            :param tokens: token sequence to decompose
            :return: decomposed token sequence
            """
            i = 0
            while i < len(tokens):
                token_type, token_val = self.vocab[tokens[i]].split('_')
                if token_type == 'BPE':
                    del tokens[i]
                    for j, to_insert in enumerate(map(int, token_val.split('-'))):
                        tokens.insert(i + j, to_insert)
                i += 1
            return tokens

        def token_types_errors(self, tokens: List[int], consider_pad: bool = False) -> float:
            """ Checks if a sequence of tokens is constituted of good token types
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

        def save_params(self, out_dir: Union[str, Path, PurePath]):
            r"""Saves the base parameters of this encoding in a txt file.
            Useful to keep track of how a dataset has been tokenized / encoded.
            It will also save the name of the class used, i.e. the encoding strategy.
            NOTE: as json can't save tuples as keys, the beat ranges are saved as strings
            with the form startingBeat_endingBeat (underscore separating these two values).

            :param out_dir: output directory to save the file
            """
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(PurePath(out_dir, 'config').with_suffix(".txt"), 'w') as outfile:
                json.dump({'pitch_range': (self.pitch_range.start, self.pitch_range.stop),
                           'beat_res': {f'{k1}_{k2}': v for (k1, k2), v in self.beat_res.items()},
                           'nb_velocities': len(self.velocities),
                           'additional_tokens': self.additional_tokens,
                           '_sos_eos': self._sos_eos,
                           '_mask': self._mask,
                           'encoding': f'{self.__class__.__name__}_bpe',
                           'token_to_event': self.vocab.token_to_event}, outfile, indent=4)

        def load_params(self, params: Union[str, Path, PurePath, Dict[str, Any]]):
            r"""Loads parameters and set the encoder attributes.

            :param params: can be a path to the parameter (json encoded) file or a dictionary
            """
            if isinstance(params, (str, Path, PurePath)):
                with open(params) as param_file:
                    params = json.load(param_file)

            if not isinstance(params['pitch_range'], range):
                params['pitch_range'] = range(*params['pitch_range'])

            for key, value in params.items():
                if key == 'beat_res':
                    value = {tuple(map(int, beat_range.split('_'))): res for beat_range, res in value.items()}
                elif key == 'additional_tokens':
                    value['TimeSignature'] = value.get('TimeSignature', False)
                elif key == 'token_to_event':
                    self.vocab = Vocabulary()
                    self.vocab._token_to_event = {int(token): event for token, event in value.items()}
                    self.vocab._event_to_token = {event: int(token) for token, event in value.items()}
                    self.vocab.update_token_types_indexes()
                    self.has_bpe = len(self.vocab.tokens_of_type('BPE')) > 0
                    continue
                setattr(self, key, value)

    # tokenizer.__class__.__bases__ += (BPE,)
    return BPE()
