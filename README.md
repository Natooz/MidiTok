# MidiTok

MidiTok is a MIDI encoding / tokenization for deep neural networks. It can process MIDI files and "tokenize" them as for text in the NLP field.

MidiTok features most known MIDI encoding strategies, and is built around the idea that they all share common parameters and methods.

## Installation

You can install MidiTok directly from pip:
```shell
pip install miditok
```
MidiTok uses MIDIToolkit, which itself uses Mido to read and write MIDI files.

## What it can do

Desc

### Common parameters

Every encoding strategy share some common parameters:

* **Pitch range:** the MIDI norm can represent pitch values from 0 to 127, but the [GM2 specification](https://www.midi.org/specifications-old/item/general-midi-2) recommend from 21 to 108 for piano, which covers the recommended pitch values for all MIDI program. Notes with pitches under or above this range can be discarded or clipped to the limits.
* **Beat resolution:** is the number of "frames" sampled within a beat. In NAME we
* **Number of velocities:** the number of velocity values you want represents. For instance if you set this parameter to 32, the velocities of the notes will be quantized into 32 velocity values from 0 to 127.
* **Additional tokens:** specify which additional tokens bringing information like chords should be included. Note that each encoding is compatible with different additional tokens.

### Encodings

Desc

### Use NAME to create your own encoding strategy

desc

## Example

```python
from miditok import REMIEncoding
from miditoolkit import MidiFile

# Parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True,
                     'Empty': True,
                     'Tempo': False,
                     'Ignore': False}

# Creates the tokenizer and loads a MIDI
remi_enc = REMIEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)
midi = MidiFile('path/to/your_midi.mid')

# Converts MIDI to tokens
tokens = remi_enc.midi_to_tokens(midi)
```

## Contributions

Desc

## Citations

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
@article{midilike2018,
    title={This time with feeling: Learning expressive musical performance},
    author={Oore, Sageev and Simon, Ian and Dieleman, Sander and Eck, Douglas and Simonyan, Karen},
    journal={Neural Computing and Applications},
    volume={32},
    number={4},
    pages={955--967},
    year={2018},
    publisher={Springer}
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



## Acknowledgments

We acknowledge [Aubay](https://www.aubay.com/index.php/language/en/home/?lang=en) and the [LIP6](https://www.lip6.fr/?LANG=en) for the financing and support of this project.
