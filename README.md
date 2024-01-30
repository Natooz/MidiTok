# MidiTok

Python package to tokenize MIDI music files, presented at the ISMIR 2021 LBDs.

![MidiTok Logo](docs/assets/logo.png?raw=true "")

[![PyPI version fury.io](https://badge.fury.io/py/miditok.svg)](https://pypi.python.org/pypi/miditok/)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/)
[![Documentation Status](https://readthedocs.org/projects/miditok/badge/?version=latest)](https://miditok.readthedocs.io/en/latest/?badge=latest)
[![GitHub CI](https://github.com/Natooz/MidiTok/actions/workflows/pytest.yml/badge.svg)](https://github.com/Natooz/MidiTok/actions/workflows/pytest.yml)
[![Codecov](https://img.shields.io/codecov/c/github/Natooz/MidiTok)](https://codecov.io/gh/Natooz/MidiTok)
[![GitHub license](https://img.shields.io/github/license/Natooz/MidiTok.svg)](https://github.com/Natooz/MidiTok/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/miditok)](https://pepy.tech/project/MidiTok)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

MidiTok can tokenize MIDI files, i.e. convert them into sequences of tokens ready to be fed to models such as Transformer, for any generation, transcription or MIR task.
MidiTok features most known [MIDI tokenizations](https://miditok.readthedocs.io/en/latest/tokenizations.html) (e.g. [REMI](https://arxiv.org/abs/2002.00212), [Compound Word](https://arxiv.org/abs/2101.02402)...), and is built around the idea that they all share common parameters and methods. It supports [Byte Pair Encoding (BPE)](https://arxiv.org/abs/2301.11975) and data augmentation.

**Documentation:** [miditok.readthedocs.com](https://miditok.readthedocs.io/en/latest/index.html)

## Install

```shell
pip install miditok
```
MidiTok uses [Symusic](https://github.com/Yikai-Liao/symusic) to read and write MIDI files, and BPE is backed by [Hugging Face 🤗tokenizers](https://github.com/huggingface/tokenizers) for super-fast encoding.

## Usage example

Below is a complete yet concise example of how you can use MidiTok. And [here](colab-notebooks/Full_Example_HuggingFace_GPT2_Transformer.ipynb) is a simple notebook example showing how to use Hugging Face models to generate music, with MidiTok taking care of tokenizing MIDIs.

```python
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator
from pathlib import Path
from symusic import Score

# Creating a multitrack tokenizer configuration, read the doc to explore other parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Loads a midi, converts to tokens, and back to a MIDI
midi = Score("path/to/your_midi.mid")
tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
converted_back_midi = tokenizer(tokens)  # PyTorch / Tensorflow / Numpy tensors supported

# Trains the tokenizer with BPE, and save it to load it back later
midi_paths = list(Path("path", "to", "midis").glob("**/*.mid"))
tokenizer.learn_bpe(vocab_size=30000, files_paths=midi_paths)
tokenizer.save_params(Path("path", "to", "save", "tokenizer.json"))
# And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
tokenizer.push_to_hub("username/model-name", private=True, token="your_hf_token")

# Creates a Dataset and a collator to be used with a PyTorch DataLoader to train a model
dataset = DatasetTok(
    files_paths=midi_paths,
    min_seq_len=100,
    max_seq_len=1024,
    tokenizer=tokenizer,
)
collator = DataCollator(
    tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"]
)
from torch.utils.data import DataLoader
data_loader = DataLoader(dataset=dataset, collate_fn=collator)
for batch in data_loader:
    print("Train your model on this batch...")
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
* [MMM](https://arxiv.org/abs/2008.06048)

You can find short presentations in the [documentation](https://miditok.readthedocs.io/en/latest/tokenizations.html).

## Contributions

Contributions are gratefully welcomed, feel free to open an issue or send a PR if you want to add a tokenization or speed up the code. You can read the [contribution guide](CONTRIBUTING.md) for details.

### Todos

* `no_duration_drums` option, discarding duration tokens for drum notes;
* Extend unimplemented additional tokens to all compatible tokenizations;
* Control Change messages;
* Speeding up the MIDI preprocess + global/track events parsing with Rust or C++ binding.

## Citation

If you use MidiTok for your research, a citation in your manuscript would be gladly appreciated. ❤️

[**[MidiTok paper]**](https://arxiv.org/abs/2310.17202)
[**[MidiTok original ISMIR publication]**](https://archives.ismir.net/ismir2021/latebreaking/000005.pdf)
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

Special thanks to all the contributors.
We acknowledge [Aubay](https://blog.aubay.com/index.php/language/en/home/?lang=en), the [LIP6](https://www.lip6.fr/?LANG=en), [LERIA](http://blog.univ-angers.fr/leria/n) and [ESEO](https://eseo.fr/en) for the initial financing and support.

## All Thanks To Our Contributors

<a href="https://github.com/Natooz/MidiTok/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Natooz/MidiTok" />
</a>
