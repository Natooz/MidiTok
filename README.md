# MidiTok

Python package to tokenize MIDI music files.

[![PyPI version fury.io](https://badge.fury.io/py/miditok.svg)](https://pypi.python.org/pypi/miditok/)
[![GitHub license](https://img.shields.io/github/license/Natooz/MidiTok.svg)](https://github.com/Natooz/MidiTok/blob/main/LICENSE)

![MidiTok Logo](https://github.com/Natooz/MidiTok/blob/assets/assets/logo.png?raw=true "")


MidiTok converts MIDI music files into sequences of tokens, i.e. integers, ready to be fed to sequential neural networks like Transformers or RNNs.
MidiTok features most known MIDI tokenization strategies, and is built around the idea that they all share common parameters and methods.

## Install

```shell
pip install miditok
```
MidiTok uses MIDIToolkit, which itself uses Mido to read and write MIDI files.

## Examples

### Tokenize a MIDI file

```python
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

# Creates the tokenizer and loads a MIDI
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens)
midi = MidiFile('path/to/your_midi.mid')

# Converts MIDI to tokens, and back to a MIDI
tokens = tokenizer.midi_to_tokens(midi)
converted_back_midi = tokenizer.tokens_to_midi(tokens, get_midi_programs(midi))

# Converts just a selected track
tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

# And convert it back (the last arg stands for (program number, is drum))
converted_back_track, tempo_changes = tokenizer.tokens_to_track(piano_tokens, midi.ticks_per_beat, (0, False))
```

### Tokenize a dataset

MidiTok will save your encoding parameters in a ```config.txt``` file to keep track of how they were converted.

```python
from miditok import REMI
from pathlib import Path

# Creates the tokenizer and list the file paths
tokenizer = REMI()  # uses defaults parameters
paths = list(Path('path', 'to', 'dataset').glob('**/*.mid'))

# A validation method to discard MIDIs we do not want
def midi_valid(midi) -> bool:
    if any(ts.numerator != 4 or ts.denominator != 4 for ts in midi.time_signature_changes):
        return False  # time signature different from 4/4
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True

# Converts MIDI files to tokens saved as JSON files
tokenizer.tokenize_midi_dataset(paths, 'path/to/save', midi_valid)
```

### Write a MIDI file from tokens

```python
from miditok import REMI
import torch

# Creates the tokenizer and list the file paths
remi_enc = REMI()  # uses defaults parameters in constants.py

# The tokens, let's say produced by your Transformer, 4 tracks of 500 tokens
tokens = torch.randint(low=0, high=len(remi_enc.vocab), size=(4, 500)).tolist()

# The instruments, here piano, violin, french horn and drums
programs = [(0, False), (41, False), (61, False), (0, True)]

# Convert to MIDI and save it
generated_midi = remi_enc.tokens_to_midi(tokens, programs)
generated_midi.dump('path/to/save/file.mid')  # could have been done above by giving the path argument
```

## Encodings

The figures represent the following music sheet as its corresponding token sequences.

![Music sheet example](https://github.com/Natooz/MidiTok/blob/assets/assets/music_sheet.png?raw=true "Music sheet example")

_Tokens are vertically stacked at index 0 from the bottom up to the top._

### MIDI-Like

Strategy used in the first symbolic music generative transformers and RNN / LSTM models. It consists of encoding the MIDI messages (Note On, Note Off, Velocity and Time Shift) into tokens as represented in a pure "MIDI way".

NOTES:
* Rests act exactly like Time-shifts. It is then recommended choosing a minimum rest range of the same first beat resolution so the time is shifted with the same accuracy. For instance if your first beat resolution is ```(0, 4): 8```, you should choose a minimum rest of ```8```.

![MIDI-Like figure](https://github.com/Natooz/MidiTok/blob/assets/assets/midi_like.png?raw=true "MIDI-Like token sequence, with Time-Shifts and Note-Off tokens")

### REMI

Proposed with the [Pop Music Transformer](https://arxiv.org/abs/2002.00212), it is a "position-based" representation. The time is represented with "_Bar_" and "_Position_" tokens that indicate respectively when a new bar is beginning, and the current position within a bar. A note is represented as a succession of a _Pitch_, _Velocity_ and _Duration_ tokens.

NOTES:
* In the original REMI paper, the tempo information are in fact the succession of two token types: a "_Token Class_" which indicate if the tempo is fast or slow, and a "_Token Value_" which represents its value with respect to the tempo class. In MidiTok we only encode one _Tempo_ token which encode its value, quantized in a number of bins set in parameters (as done for velocities).
* Including tempo tokens in a multitrack task with REMI is not recommended. Generating several tracks would lead to multiple and ambiguous tempo changes. So in MidiTok only the tempo changes of the first track will be kept in the final created MIDI.
* Position tokens are always following Rest tokens to make sure the position of the following notes are explicitly stated. Bar tokens can follow Rest tokens depending on their respective value and your parameters.

![REMI figure](https://github.com/Natooz/MidiTok/blob/assets/assets/remi.png?raw=true "REMI sequence, time is tracked with Bar and position tokens")

### Compound Word

Introduced with the [Compound Word Transformer](https://arxiv.org/abs/2101.02402) this representation is similar to the REMI encoding. The key difference is that tokens of different types of a same "event" are combined and processed at the same time by the model.
_Pitch_, _Velocity_ and _Durations_ tokens of a same note will be combined for instance. The greatest benefit of this encoding strategy is the **reduced sequence lengths** that it creates, which means less time and memory consumption as transformers (with softmax attention) have a quadratic complexity.

You can combine them in your model the way you want. CP Word authors concatenated each embeddings and projected the sequence with a projection matrix, resulting in a _d_-dimensional vector (_d_ being the model size).

At decoding, the easiest way to predict multiple tokens (employed by the original authors) is to project the output vector of your model with several projection matrices, one for each token type.

![Compound Word figure](https://github.com/Natooz/MidiTok/blob/assets/assets/cp_word.png?raw=true "CP Word sequence, tokens of the same family are grouped together")

### Structured

Presented with the [Piano Inpainting Application](https://arxiv.org/abs/2107.05944), it is similar to the MIDI-Like encoding but with _Duration_ tokens instead Note-Off.
The main advantage of this encoding is the consistent token type transitions it imposes, which can greatly speed up training. The structure is as: _Pitch_ -> _Velocity_ -> _Duration_ -> _Time Shift_ -> ... (pitch again)
To keep this property, no additional token can be inserted in MidiTok's implementation.

![Structured figure](https://github.com/Natooz/MidiTok/blob/assets/assets/structured.png?raw=true "Structured MIDI encoding, the token types always follow the same transition pattern")

### Octuple

Introduced with [Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/abs/2106.05630). Each note of each track is the combination of multiple embeddings: _Pitch_, _Velocity_, _Duration_, _Track_, current _Bar_, current _Position_ and additional tokens.
The main benefit is the reduction of the sequence lengths, its multitrack capabilities, and its simple structure easy to decode.
The Bar and Position embeddings can act as a positional encoding, but the authors of the original paper still applied a token-wise positional encoding afterward.

NOTES:
* In MidiTok, the tokens are first sorted by time, then track, then pitch values.
* This implementation uses _Program_ tokens to distinguish tracks, on their MIDI program. Hence, two tracks with the same program will be treated as being the same.
* Time signature and Tempo tokens are optional in MidiTok, you can chose to use them or not with the ```additional_tokens``` parameter.
* [Octuple Mono](miditok/octuple_mono.py) is a modified version with no program embedding at each time step.

![Octuple figure](https://github.com/Natooz/MidiTok/blob/assets/assets/octuple.png?raw=true "Octuple sequence, with a bar and position embeddings")

### MuMIDI

Presented with the [PopMAG](https://arxiv.org/abs/2008.07703) model, this representation is mostly suited for multitrack tasks. The time is based on _Position_ and _Bar_ tokens as REMI and Compound Word.
The key idea of MuMIDI is to represent every track in a single sequence. At each time step, "_Track_" tokens preceding note tokens indicate from which track they are.
MuMIDI also include a "built-in" positional encoding mechanism. At each time step, embeddings of the current bar and current position are merged with the token. For a note, the _Pitch_, _Velocity_ and _Duration_ embeddings are also merged together.

NOTES:
* In MidiTok, the tokens are first sorted by time, then track, then pitch values.
* In the original MuMIDI, _Chord_ tokens are placed before Track tokens. We decided in MidiTok to put them after as chords are produced by one instrument, and several instruments can produce more than one chord at a time step.
* This implementation uses _Program_ tokens to distinguish tracks, on their MIDI program. Hence, two tracks with the same program will be treated as being the same.
* As in the original MuMIDI implementation, MidiTok distinguishes pitch tokens of drums from pitch tokens of other instruments. More details in the [code](miditok/mumidi.py).

![MuMIDI figure](https://github.com/Natooz/MidiTok/blob/assets/assets/mumidi.png?raw=true "MuMIDI sequence, with a bar and position embeddings")

### Create your own

You can easily create your own encoding strategy and benefit from the MidiTok framework. Just create a class inheriting from the [MIDITokenizer](miditok/midi_tokenizer_base.py) base class, and override the ```track_to_tokens```, ```tokens_to_track```,  ```_create_vocabulary``` and ```_create_token_types_graph``` methods with your tokenization strategy.

We encourage you to read the docstring of the [Vocabulary class](miditok/vocabulary.py) to learn how to use it for your strategy.

## Features

### Common parameters

Every encoding strategy share some common parameters around which the tokenizers are built:

* **Pitch range:** the MIDI norm can represent pitch values from 0 to 127, but the [GM2 specification](https://www.midi.org/specifications-old/item/general-midi-2) recommend from 21 to 108 for piano, which covers the recommended pitch values for all MIDI program. Notes with pitches under or above this range can be discarded or clipped to the limits.
* **Beat resolution:** is the number of samples within a beat. MidiTok handles this with a flexible way: a dictionary of the form ```{(0, 4): 8, (3, 8): 4, ...}```. The keys are tuples indicating a range of beats, ex 0 to 4 for the first bar. The values are the resolutions, in samples per beat, of the given range, here 8 for the first. This way you can create a tokenizer with durations / time shifts of different lengths and resolutions.
* **Number of velocities:** the number of velocity values you want represents. For instance if you set this parameter to 32, the velocities of the notes will be quantized into 32 velocity values from 0 to 127.
* **Additional tokens:** specify which additional tokens bringing information like chords should be included. Note that each encoding is compatible with different additional tokens.

Check [constants.py](miditok/constants.py) to see how these parameters are constructed.

### Additional tokens

MidiTok offers the possibility to insert additional tokens in the encodings.
These tokens bring additional information about the structure and content of MIDI tracks to explicitly use them to train a neural network.

* **Chords:** indicates the presence of a chord at a certain time step. MidiTok uses a chord detection method based on onset times and duration. This allows MidiTok to detect precisely chords without ambiguity, whereas most chord detection methods in symbolic music based on chroma features can't.
* **Rests:** includes "Rest" events whenever a segment of time is silent, i.e. no note is played within. This token type is decoded as a "Time-Shift" event, meaning the time will be shifted according to its value. You can choose the minimum and maximum rests values to represent (default is 1/2 beat to 8 beats). Note that rests shorter than one beat are only divisible by the first beat resolution, e.g. a rest of 5/8th of a beat will be a succession of ```Rest_0.4``` and ```Rest_0.1```, where the first number indicate the rest duration in beats and the second in samples / positions.
* **Tempos:** specifies the current tempo. This allows to train a model to predict tempo changes alongside with the notes, unless specified in the chart below. Tempo values are quantized on your specified range and number (default is 32 tempos from 40 to 250).
* **Programs:** used to specify an instrument / MIDI program. MidiTok only offers the possibility to include these tokens in the vocabulary for you, but won't use them. If you need model multitrack symbolic music with other methods than Octuple / MuMIDI, MidiTok leaves you the choice / task to represent the track information the way you want. You can do it as in [LakhNES](https://github.com/chrisdonahue/LakhNES) or [MMM](https://metacreation.net/mmm-multi-track-music-machine/).
* **Time Signature:** specifies the current time signature. Only implemented with Octuple in MidiTok a.t.w.

Additionally, MidiTok offers to include *Program* tokens in the vocabulary of MIDI-Like, REMI and CP Word.
We do not consider them additional tokens though as they are not used anywhere in MidiTok, but intended for you to insert them at the beginning of each sequence as *Start Of Sequence* tokens.

|                |   MIDI-Like   | REMI          | Compound Word | Structured |    Octuple    |    MuMIDI     |
|----------------|:-------------:|:--------------:|:--------------:|:--------:|:-------------:|:-------------:|
| Chord          |       ✅       | ✅             | ✅             | ❌        |       ❌       |       ✅       |
| Rest           |       ✅       | ✅             | ✅             | ❌        |       ❌       |       ❌       |
| Tempo          | ✅<sup>1</sup> | ✅<sup>1</sup> | ✅<sup>1</sup> | ❌        |       ✅       |       ✅       |
| Program        | ✅             | ✅             | ✅             | ✅        | ✅<sup>3</sup> | ✅<sup>3</sup> |
| Time Signature |       ❌       | ❌             | ❌             | ❌        |      ✅        |      ❌        |

<sup>1</sup> Should not be used with multiple tracks. Otherwise, at decoding, only the events of the first track will be considered.\
<sup>2</sup> Only used in the input as additional information. At decoding no tempo tokens should be predicted, i.e will be considered.\
<sup>3</sup> Integrated by default.

## Limitations

For the tokenization methods using Bar tokens (REMI, Compound Word and MuMIDI), **MidiTok only considers a 4/x time signature** for now. This means that each bar is considered covering 4 beats.

Future updates will support other time signatures, and time signature changes for compatible tokenizations.

## Contributions

Contributions are gratefully welcomed, feel free to send a PR if you want to add an encoding strategy or speed up the code. Just make sure to pass the [tests](tests).

## Todo

* Time Signature
* Allow switching between beat / second time unit
* Control Change messages
* Automatic data augmentation on pitch

## Citations

### MidiTok:

[**Paper**](https://archives.ismir.net/ismir2021/latebreaking/000005.pdf) | [**Poster**](https://s3.eu-west-1.amazonaws.com/production-main-contentbucket52d4b12c-1x4mwd6yn8qjn/8672541d-e891-41ba-9444-26f31b038ee6.pdf)
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

We acknowledge [Aubay](https://www.aubay.com/index.php/language/en/home/?lang=en), the [LIP6](https://www.lip6.fr/?LANG=en), [LERIA](http://blog.univ-angers.fr/leria/n) and [ESEO](https://eseo.fr/en) for the financing and support of this project.
Special thanks to all the contributors.
