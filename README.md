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

![MidiTok Logo](https://github.com/Natooz/MidiTok/blob/assets/assets/logo.png?raw=true "")


MidiTok converts MIDI music files into sequences of tokens, i.e. integers, ready to be fed to sequential neural networks like Transformers or RNNs.
MidiTok features most known MIDI tokenization strategies, and is built around the idea that they all share common parameters and methods. It contains methods allowing to properly pre-process any MIDI file, and also supports Byte Pair Encoding (BPE).

## Install

```shell
pip install miditok
```
MidiTok uses [MIDIToolkit](https://github.com/YatingMusic/miditoolkit), which itself uses [Mido](https://github.com/mido/mido) to read and write MIDI files.

## Examples

### Tokenize a MIDI file

```python
from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

# Creates the tokenizer and loads a MIDI
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
midi = MidiFile('path/to/your_midi.mid')

# Converts MIDI to tokens, and back to a MIDI
tokens = tokenizer(midi)  # automatically detects MIDIs and tokens before converting
converted_back_midi = tokenizer(tokens, get_midi_programs(midi))

# Converts just a selected track
tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

# And convert it back (the last arg stands for (program number, is drum))
converted_back_track, tempo_changes = tokenizer.tokens_to_track(piano_tokens, midi.ticks_per_beat, (0, False))
```

### Tokenize a dataset, perform data augmentation and apply Byte Pair Encoding

MidiTok will save your encoding parameters in a ```config.txt``` file to keep track of how they were converted.

```python
from miditok import REMI
from pathlib import Path

# Creates the tokenizer and list the file paths
tokenizer = REMI(mask=True)  # using defaults parameters (constants.py)
midi_paths = list(Path('path', 'to', 'dataset').glob('**/*.mid'))

# A validation method to discard MIDIs we do not want
# It can also be used for custom pre-processing, for instance if you want to merge
# some tracks before tokenizing a MIDI file
def midi_valid(midi) -> bool:
    if any(ts.numerator != 4 for ts in midi.time_signature_changes):
        return False  # time signature different from 4/*, 4 beats per bar
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True

# Converts MIDI files to tokens saved as JSON files
data_augmentation_offsets = [2, 2, 1]  # perform data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path('path', 'to', 'tokens_noBPE'), midi_valid, data_augmentation_offsets)

# Constructs the vocabulary with BPE
tokenizer.learn_bpe(tokens_path=Path('path', 'to', 'tokens_noBPE'), vocab_size=500,
                    out_dir=Path('path', 'to', 'tokens_BPE'), files_lim=300)

# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(Path('path', 'to', 'tokens_noBPE'), Path('path', 'to', 'tokens_BPE'))
```

### Write a MIDI file from tokens

```python
from miditok import REMI
import torch

# Creates the tokenizer
tokenizer = REMI()  # using defaults parameters (constants.py)

# The tokens, let's say produced by your Transformer, 4 tracks of 500 tokens
tokens = torch.randint(low=0, high=len(tokenizer.vocab), size=(4, 500)).tolist()

# The instruments, here piano, violin, french horn and drums
programs = [(0, False), (41, False), (61, False), (0, True)]

# Convert to MIDI and save it
generated_midi = tokenizer.tokens_to_midi(tokens, programs)
generated_midi.dump('path/to/save/file.mid')  # could have been done above by giving the path argument
```

## Tokenizations

The figures represent the following music sheet as its corresponding token sequences.

![Music sheet example](https://github.com/Natooz/MidiTok/blob/assets/assets/music_sheet.png?raw=true "Music sheet example")

_Tokens are vertically stacked at index 0 from the bottom up to the top._

### MIDI-Like

Strategy used in the first symbolic music generative transformers and RNN / LSTM models. MIDI messages (Note On, Note Off, Velocity and Time Shift) are represented as tokens.


![MIDI-Like figure](https://github.com/Natooz/MidiTok/blob/assets/assets/midi_like.png?raw=true "MIDI-Like token sequence, with TimeShift and NoteOff tokens")

### TimeShift Duration (TSD)

A strategy similar to MIDI-Like, but uses explicit `Duration` tokens to represent note durations, which have showed [better results](https://arxiv.org/abs/2002.00212), helping models to learn better.

![MIDI-Like figure](https://github.com/Natooz/MidiTok/blob/assets/assets/tsd.png?raw=true "MIDI-Like token sequence, with TimeShifts and NoteOff tokens")

### REMI

Proposed with the [Pop Music Transformer](https://arxiv.org/abs/2002.00212), it is a "position-based" representation. The time is represented with "_Bar_" and "_Position_" tokens that indicate respectively when a new bar is beginning, and the current position within a bar. A note is represented as a succession of a _Pitch_, _Velocity_ and _Duration_ tokens.

NOTE:
* In the original REMI paper, the tempo information are in fact the succession of two token types: a "_Token Class_" which indicate if the tempo is fast or slow, and a "_Token Value_" which represents its value with respect to the tempo class. In MidiTok we only encode one _Tempo_ token which encode its value, quantized in a number of bins set in parameters (as done for velocities).

![REMI figure](https://github.com/Natooz/MidiTok/blob/assets/assets/remi.png?raw=true "REMI sequence, time is tracked with Bar and position tokens")

### Compound Word

Introduced with the [Compound Word Transformer](https://arxiv.org/abs/2101.02402), this tokenization is similar to REMI but uses embedding pooling operations to reduce the overall sequence length: some tokens are first converted to embeddings by a model, them merged / pooled into a single one.
_Pitch_, _Velocity_ and _Durations_ tokens of a same note will be combined. The **sequence length reduction** means less time and memory complexity.

For generation tasks, the decoding implies to project the last hidden states to several output layers, one for each token type, and samples from the multiple output distributions.

![Compound Word figure](https://github.com/Natooz/MidiTok/blob/assets/assets/cp_word.png?raw=true "CP Word sequence, tokens of the same family are grouped together")

### Structured

Presented with the [Piano Inpainting Application](https://arxiv.org/abs/2107.05944), it is similar to MIDI-Like but with _Duration_ tokens instead NoteOff.
Its main advantage is the consistent token type transitions it imposes, which can greatly speed up training. The structure is as: _Pitch_ -> _Velocity_ -> _Duration_ -> _Time Shift_ -> ... (pitch again)
To keep this property, no additional token can be inserted in MidiTok's implementation.

![Structured figure](https://github.com/Natooz/MidiTok/blob/assets/assets/structured.png?raw=true "Structured MIDI encoding, the token types always follow the same transition pattern")

### Octuple

Introduced with [Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/abs/2106.05630). Each note of each track is the combination of multiple embeddings: _Pitch_, _Velocity_, _Duration_, _Track_, current _Bar_, current _Position_ and additional tokens.
Its considerably reduces the sequence lengths, while handling multitrack. Generating with it requires however to sample from several distributions and can be delicate. This tokenization is best suited for MIR and classification tasks.
The Bar and Position embeddings can act as a positional encoding, but the authors of the original paper still applied a token-wise positional encoding afterward.

NOTES:
* In MidiTok, the tokens are first sorted by time, then track, then pitch values.
* This implementation uses _Program_ tokens to distinguish tracks, on their MIDI program. Hence, two tracks with the same program will be treated as being the same.
* Time signature and Tempo tokens are optional, you can choose to use them or not with the ```additional_tokens``` parameter.
* [Octuple Mono](miditok/octuple_mono.py) is a modified version with no program embedding at each time step.

![Octuple figure](https://github.com/Natooz/MidiTok/blob/assets/assets/octuple.png?raw=true "Octuple sequence, with a bar and position embeddings")

### MuMIDI

Presented with the [PopMAG](https://arxiv.org/abs/2008.07703) model, this tokenization made for multitrack tasks and uses embedding pooling. The time is based on _Position_ and _Bar_ tokens as REMI and Compound Word.
The key idea of MuMIDI is to represent every track in a single sequence. At each time step, "_Track_" tokens preceding note tokens indicate from which track they are. Generating with it requires however to sample from several distributions and can be delicate.
MuMIDI also include a "built-in" positional encoding mechanism. As in the original paper, the pitches of drums are distinct from those of all other instruments.

NOTES:
* In MidiTok, the tokens are first sorted by time, then track, then pitch values.
* This implementation uses _Program_ tokens to distinguish tracks, on their MIDI program. Hence, two tracks with the same program will be treated as being the same.

![MuMIDI figure](https://github.com/Natooz/MidiTok/blob/assets/assets/mumidi.png?raw=true "MuMIDI sequence, with a bar and position embeddings")

### Create your own

You can easily create your own tokenization and benefit from the MidiTok framework. Just create a class inheriting from the [MIDITokenizer](miditok/midi_tokenizer_base.py) base class, and override the ```track_to_tokens```, ```tokens_to_track```,  ```_create_vocabulary``` and ```_create_token_types_graph``` methods with your tokenization strategy.

We encourage you to read the docstring of the [Vocabulary class](miditok/vocabulary.py) to learn how to use it for your strategy.

## Features

### Common parameters

Every tokenization share some common parameters around which the tokenizers are built:

* **Pitch range:** the MIDI norm can represent pitch values from 0 to 127, but the [GM2 specification](https://www.midi.org/specifications-old/item/general-midi-2) recommend from 21 to 108 for piano, which covers the recommended pitch values for all MIDI program. Notes with pitches under or above this range can be discarded or clipped to the limits.
* **Beat resolution:** the number of samples within a beat. MidiTok handles this with a flexible way: a dictionary of the form `{(0, 4): 8, (3, 8): 4, ...}`. The keys are tuples indicating a range of beats, ex 0 to 4 for the first bar. The values are the resolutions, in samples per beat, of the given range, here 8 for the first. This way you can create a tokenizer with durations / time shifts of different lengths and resolutions.
* **Number of velocities:** the number of velocity values to represent. For instance with 32, the velocities of the notes will be quantized into 32 velocity values from 0 to 127.
* **Additional tokens:** specify which additional tokens should be included. Note that each encoding is compatible with different additional tokens.

Check [constants.py](miditok/constants.py) to see how these parameters are constructed.

### Byte Pair Encoding (BPE)

[BPE](https://www.derczynski.com/papers/archive/BPE_Gage.pdf) is a compression technique that originally combines the most recurrent byte pairs in a corpus.
In the context of tokens, it allows to combine the most recurrent token successions by replacing them with a new created symbol (token). This naturally increases the size of the vocabulary, while reducing the overall sequence length.
Today BPE is used to build almost all tokenizations of natural language, as [it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units](https://aclanthology.org/P16-1162/).
You can apply it to symbolic music with MidiTok, by first learning the vocabulary (`tokenizer.learn_bpe()`), and then convert a dataset with BPE (`tokenizer.apply_bpe_to_dataset()`).
All tokenizations not based on embedding pooling are compatible!

### Special tokens

When creating a tokenizer, you can specify to include some special tokens in its vocabulary, by giving the arguments:

* **`pad`** (default `True`) --> `PAD_None`: a padding token to use when training a model with batches of sequences of unequal lengths. The padding token will be at index 0 of the vocabulary.
* **`sos_eos`** (default `False`) --> `SOS_None` and `EOS_None`: "Start Of Sequence" and "End Of Sequence" tokens, designed to be placed respectively at the beginning and end of a token sequence during training. At inference, the EOS token tells when to end the generation.
* **`mask`** (default `False`) --> `MASK_None`: a masking token, to use when pre-training a (bidirectional) model with a self-supervised objective like [BERT](https://arxiv.org/abs/1810.04805).

### Additional tokens

MidiTok offers the possibility to insert additional tokens in the encodings.
These tokens bring additional information about the structure and content of MIDI tracks to explicitly use them to train a neural network.

* **Chords:** indicates the presence of a chord at a certain time step. MidiTok uses a chord detection method based on onset times and duration. This allows MidiTok to detect precisely chords without ambiguity, whereas most chord detection methods in symbolic music based on chroma features can't.
* **Rests:** includes "Rest" events whenever a segment of time is silent, i.e. no note is played within. This token type is decoded as a "TimeShift" event, meaning the time will be shifted according to its value. You can choose the minimum and maximum rests values to represent (default is 1/2 beat to 8 beats). Note that rests shorter than one beat are only divisible by the first beat resolution, e.g. a rest of 5/8th of a beat will be a succession of ```Rest_0.4``` and ```Rest_0.1```, where the first number indicate the rest duration in beats and the second in samples / positions.
* **Tempos:** specifies the current tempo. This allows to train a model to predict tempo changes alongside with the notes, unless specified in the chart below. Tempo values are quantized on your specified range and number (default is 32 tempos from 40 to 250).
* **Programs:** used to specify an instrument / MIDI program. MidiTok only offers the possibility to include these tokens in the vocabulary for you, but won't use them. If you need model multitrack symbolic music with other methods than Octuple / MuMIDI, MidiTok leaves you the choice / task to represent the track information the way you want. You can do it as in [LakhNES](https://github.com/chrisdonahue/LakhNES) or [MMM](https://metacreation.net/mmm-multi-track-music-machine/).
* **Time Signature:** specifies the current time signature. Only implemented with Octuple in MidiTok a.t.w.

| Token type     |   MIDI-Like   |      TSD      |     REMI      | Compound Word | Structured |    Octuple    |    MuMIDI     |
|----------------|:-------------:|:-------------:|:-------------:|:-------------:|:----------:|:-------------:|:-------------:|
| Chord          |       ✅       |       ✅       |       ✅       |       ✅       |     ❌      |       ❌       | ✅<sup>3</sup> |
| Rest           |       ✅       |       ✅       | ✅<sup>2</sup> | ✅<sup>2</sup> |     ❌      |       ❌       |       ❌       |
| Tempo          | ✅<sup>1</sup> | ✅<sup>1</sup> | ✅<sup>1</sup> | ✅<sup>1</sup> |     ❌      |       ✅       |       ✅       |
| Program        |       ✅       |       ✅       |       ✅       |       ✅       |     ✅      | ✅<sup>5</sup> | ✅<sup>5</sup> |
| Time signature |       ❌       |       ❌       |       ❌       |       ❌       |     ❌      |       ✅       |       ❌       |

<sup>1</sup> Should not be used with multiple tracks. Otherwise, at decoding, only the events of the first track will be considered.\
<sup>2</sup> Position tokens are always following Rest tokens to make sure the position of the following notes are explicitly stated. Bar tokens can follow Rest tokens depending on their respective value and your parameters.\
<sup>3</sup> In the original MuMIDI paper, _Chord_ tokens are placed before Track tokens. We decided in MidiTok to put them after as chords are produced by one instrument, and several instruments can produce more than one chord at a time step.\
<sup>4</sup> Integrated by default.

## Limitations

Tokenizations using Bar tokens (REMI, Compound Word and MuMIDI) **only considers a 4/x time signature** for now. This means that each bar is considered covering 4 beats.

## Contributions

Contributions are gratefully welcomed, feel free to open an issue or send a PR if you want to add a tokenization or speed up the code. Just make sure to pass the [tests](tests).

## Todo

* Time Signature
* Control Change messages
* Data augmentation on duration values at the MIDI level
* [WIP] Documentation website

## Citations

### MidiTok:

[**Paper**](https://archives.ismir.net/ismir2021/latebreaking/000005.pdf)
```bibtex
@inproceedings{miditok2021,
    title={MidiTok: A Python package for MIDI file tokenization},
    author={Nathan Fradet, Jean-Pierre Briot, Fabien Chhel, Amal El Fallah Seghrouchni, Nicolas Gutowski},
    booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference},
    year={2021}
}
```

### Tokenizations:

```bibtex
@article{midilike2018,
    title={This time with feeling: Learning expressive musical performance},
    author={Oore, Sageev and Simon, Ian and Dieleman, Sander and Eck, Douglas and Simonyan, Karen},
    journal={Neural Computing and Applications},
    year={2018},
    publisher={Springer}
}
```

```bibtex
@inproceedings{remi2020,
    title={Pop Music Transformer: Beat-based modeling and generation of expressive Pop piano compositions},
    author={Huang, Yu-Siang and Yang, Yi-Hsuan},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
    year={2020}
}
```

```bibtex
@inproceedings{cpword2021,
    title={Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs},
    author={Hsiao, Wen-Yi and Liu, Jen-Yu and Yeh, Yin-Cheng and Yang, Yi-Hsuan},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
}
```

```bibtex
@misc{structured2021,
    title={The Piano Inpainting Application},
    author={Gaëtan Hadjeres and Léopold Crestel},
    year={2021},
    eprint={2107.05944},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```

```bibtex
@inproceedings{mumidi2020,
    author = {Ren, Yi and He, Jinzheng and Tan, Xu and Qin, Tao and Zhao, Zhou and Liu, Tie-Yan},
    title = {PopMAG: Pop Music Accompaniment Generation},
    year = {2020},
    publisher = {Association for Computing Machinery},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia}
}
```

```bibtex
@misc{octuple2021,
    title={MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training}, 
    author={Mingliang Zeng and Xu Tan and Rui Wang and Zeqian Ju and Tao Qin and Tie-Yan Liu},
    year={2021},
    eprint={2106.05630},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```



## Acknowledgments

We acknowledge [Aubay](https://blog.aubay.com/index.php/language/en/home/?lang=en), the [LIP6](https://www.lip6.fr/?LANG=en), [LERIA](http://blog.univ-angers.fr/leria/n) and [ESEO](https://eseo.fr/en) for the financing and support of this project.
Special thanks to all the contributors.
