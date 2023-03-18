# MidiTok

Python package to tokenize MIDI music files, presented at the ISMIR 2021 LBD.

[![PyPI version fury.io](https://badge.fury.io/py/miditok.svg)](https://pypi.python.org/pypi/miditok/)
[![Python 3.7](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/)
[![Documentation Status](https://readthedocs.org/projects/miditok/badge/?version=latest)](https://miditok.readthedocs.io/en/latest/?badge=latest)
![GitHub CI](https://github.com/Natooz/MidiTok/actions/workflows/pytest.yml/badge.svg)
[![Codecov](https://img.shields.io/codecov/c/github/Natooz/MidiTok)](https://codecov.io/gh/Natooz/MidiTok)
[![GitHub license](https://img.shields.io/github/license/Natooz/MidiTok.svg)](https://github.com/Natooz/MidiTok/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/MidiTok)](https://pepy.tech/project/MidiTok)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![MidiTok Logo](docs/assets/logo.png?raw=true "")


MidiTok converts MIDI music files into sequences of tokens, ready to be fed to sequential deep learning models like Transformers.
MidiTok features most known [MIDI tokenizations](https://miditok.readthedocs.io/en/latest/tokenizations.html) (e.g. [REMI](https://arxiv.org/abs/2002.00212), [Compound Word](https://arxiv.org/abs/2101.02402)...), and is built around the idea that they all share common parameters and methods. It contains methods allowing to properly pre-process any MIDI file, and also supports Byte Pair Encoding (BPE).

**Documentation:** [miditok.readthedocs.com](https://miditok.readthedocs.io/en/latest/index.html)

## Install

```shell
pip install miditok
```
MidiTok uses [MIDIToolkit](https://github.com/YatingMusic/miditoolkit), which itself uses [Mido](https://github.com/mido/mido) to read and write MIDI files, and BPE is backed by [Hugging Face 🤗tokenizers](https://github.com/huggingface/tokenizers) for super fast encoding.

## Usage example

The most basic and useful methods are summarized here.

```python
from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path

# Creates the tokenizer and loads a MIDI
tokenizer = REMI()  # using the default parameters, read the documentation to customize your tokenizer
midi = MidiFile('path/to/your_midi.mid')

# Converts MIDI to tokens, and back to a MIDI
tokens = tokenizer(midi)  # automatically detects MIDIs and tokens before converting
converted_back_midi = tokenizer(tokens, get_midi_programs(midi))  # PyTorch / Tensorflow / Numpy tensors supported

# Converts MIDI files to tokens saved as JSON files
midi_paths = list(Path('path', 'to', 'dataset').glob('**/*.mid'))
data_augmentation_offsets = [2, 2, 1]  # data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path('path', 'to', 'tokens_noBPE'),
                                data_augment_offsets=data_augmentation_offsets)

# Constructs the vocabulary with BPE, from the tokenized files
tokenizer.learn_bpe(
    vocab_size=500,
    tokens_paths=list(Path('path', 'to', 'tokens_noBPE').glob("**/*.json")),
    start_from_empty_voc=False,
)

# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(Path('path', 'to', 'tokens_noBPE'), Path('path', 'to', 'tokens_BPE'))
```

## Tokenizations

MidiTok implements the tokenizations: (links to original papers)
* [REMI](https://dl.acm.org/doi/10.1145/3394171.3413671)
* [REMI+](https://openreview.net/forum?id=NyR8OZFHw6i)
* [MIDI-Like](https://link.springer.com/article/10.1007/s00521-018-3758-9)
* [TSD](https://arxiv.org/abs/2301.11975)
* [Structured](https://arxiv.org/abs/2107.05944)
* [CPWord](https://ojs.aaai.org/index.php/AAAI/article/view/16091)
* [Octuple](https://aclanthology.org/2021.findings-acl.70)
* [MuMIDI](https://dl.acm.org/doi/10.1145/3394171.3413721)

You can find short presentations in the [documentation](https://miditok.readthedocs.io/en/latest/tokenizations.html).

## Limitations

Tokenizations using Bar tokens (REMI, Compound Word and MuMIDI) **only considers a 4/x time signature** for now. This means that each bar is considered covering 4 beats.

## Contributions

Contributions are gratefully welcomed, feel free to open an issue or send a PR if you want to add a tokenization or speed up the code. Just make sure your modifications pass the [tests](tests), and format your code with [black](https://github.com/psf/black).

### Todo

* Time Signature
* Control Change messages
* Option to represent pitch values as pitch intervals, as [it seems to improve performances](https://ismir2022program.ismir.net/lbd_369.html).
* Speeding up MIDI read / load (Rust / C++ binding)
* Data augmentation on duration values at the MIDI level

## Citation

If you use MidiTok for your research, a citation in your manuscript would be gladly appreciated. ❤️

[**MidiTok paper**](https://archives.ismir.net/ismir2021/latebreaking/000005.pdf)
```bibtex
@inproceedings{miditok2021,
    title={{MidiTok}: A Python package for {MIDI} file tokenization},
    author={Fradet, Nathan and Briot, Jean-Pierre and Chhel, Fabien and El Fallah Seghrouchni, Amal and Gutowski, Nicolas},
    booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference},
    year={2021},
    url={https://archives.ismir.net/ismir2021/latebreaking/000005.pdf},
}
```

The BibTeX citations of all tokenizations can be found [in the documentation](https://miditok.readthedocs.io/en/latest/citations.html)


## Acknowledgments

We acknowledge [Aubay](https://blog.aubay.com/index.php/language/en/home/?lang=en), the [LIP6](https://www.lip6.fr/?LANG=en), [LERIA](http://blog.univ-angers.fr/leria/n) and [ESEO](https://eseo.fr/en) for the financing and support of this project.
Special thanks to all the contributors.
