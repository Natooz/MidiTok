# MidiTok

Python package to tokenize music files, introduced at the ISMIR 2021 LBDs.

![MidiTok Logo](docs/assets/miditok_logo_stroke.png?raw=true "")

[![PyPI version fury.io](https://badge.fury.io/py/miditok.svg)](https://pypi.python.org/pypi/miditok/)
[![Python 3.9](https://img.shields.io/badge/python-≥3.9-blue.svg)](https://www.python.org/downloads/release/)
[![Documentation Status](https://readthedocs.org/projects/miditok/badge/?version=latest)](https://miditok.readthedocs.io/en/latest/?badge=latest)
[![GitHub CI](https://github.com/Natooz/MidiTok/actions/workflows/pytest.yml/badge.svg)](https://github.com/Natooz/MidiTok/actions/workflows/pytest.yml)
[![Codecov](https://img.shields.io/codecov/c/github/Natooz/MidiTok)](https://codecov.io/gh/Natooz/MidiTok)
[![GitHub license](https://img.shields.io/github/license/Natooz/MidiTok.svg)](https://github.com/Natooz/MidiTok/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/miditok)](https://pepy.tech/project/MidiTok)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

MidiTok can tokenize MIDI and abc files, i.e. convert them into sequences of tokens ready to be fed to models such as Transformer, for any generation, transcription or MIR task.
MidiTok features most known [music tokenizations](https://miditok.readthedocs.io/en/latest/tokenizations.html) (e.g. [REMI](https://arxiv.org/abs/2002.00212), [Compound Word](https://arxiv.org/abs/2101.02402)...), and is built around the idea that they all share common parameters and methods. Tokenizers can be trained with [Byte Pair Encoding (BPE)](https://aclanthology.org/2023.emnlp-main.123/), [Unigram](https://aclanthology.org/P18-1007/) and [WordPiece](https://arxiv.org/abs/1609.08144), and it offers data augmentation methods.

MidiTok is integrated with the Hugging Face Hub 🤗! Don't hesitate to share your models to the community!

**Documentation:** [miditok.readthedocs.com](https://miditok.readthedocs.io/en/latest/index.html)

## Install

```shell
pip install miditok
```
MidiTok uses [Symusic](https://github.com/Yikai-Liao/symusic) to read and write MIDI and abc files, and BPE/Unigram is backed by [Hugging Face 🤗tokenizers](https://github.com/huggingface/tokenizers) for superfast encoding.

## Usage example

Tokenizing and detokenzing can be done by calling the tokenizer:

```python
from miditok import REMI, TokenizerConfig
from symusic import Score

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Loads a midi, converts to tokens, and back to a MIDI
midi = Score("path/to/your_midi.mid")
tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
converted_back_midi = tokenizer(tokens)  # PyTorch, Tensorflow and Numpy tensors are supported
```

Here is a complete yet concise example of how you can use MidiTok to train any PyTorch model. And [here](colab-notebooks/Example_HuggingFace_Mistral_Transformer.ipynb) is a simple notebook example showing how to use Hugging Face models to generate music, with MidiTok taking care of tokenizing music files.

```python
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Train the tokenizer with Byte Pair Encoding (BPE)
files_paths = list(Path("path", "to", "midis").glob("**/*.mid"))
tokenizer.train(vocab_size=30000, files_paths=files_paths)
tokenizer.save(Path("path", "to", "save", "tokenizer.json"))
# And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
tokenizer.push_to_hub("username/model-name", private=True, token="your_hf_token")

# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path("path", "to", "midi_chunks")
split_files_for_training(
    files_paths=files_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=1024,
)

# Create a Dataset, a DataLoader and a collator to train a model
dataset = DatasetMIDI(
    files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

# Iterate over the dataloader to train a model
for batch in dataloader:
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
* [PerTok](https://www.arxiv.org/abs/2410.02060)

You can find short presentations in the [documentation](https://miditok.readthedocs.io/en/latest/tokenizations.html).

## Contributions

Contributions are gratefully welcomed, feel free to open an issue or send a PR if you want to add a tokenization or speed up the code. You can read the [contribution guide](CONTRIBUTING.md) for details.

### Todos

* Support music-xml files;
* `no_duration_drums` option, discarding duration tokens for drum notes;
* Control Change messages;
* Speed-up global/track events parsing with Rust or C++ bindings.

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

@Natooz thanks its employers who allowed him to develop this project, by chronological order [Aubay](https://blog.aubay.com/index.php/language/en/home/?lang=en), the [LIP6 (Sorbonne University)](https://www.lip6.fr/?LANG=en), and the [Metacreation Lab (Simon Fraser University)](https://www.metacreation.net).

## All Thanks To Our Contributors

<a href="https://github.com/Natooz/MidiTok/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Natooz/MidiTok" />
</a>
