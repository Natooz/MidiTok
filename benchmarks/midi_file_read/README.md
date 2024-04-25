# MIDI files reading

This benchmark measures the read times of MIDI files, comparing the [symusic](https://github.com/Yikai-Liao/symusic), [miditoolkit](https://github.com/YatingMusic/miditoolkit) and [pretty_midi](https://github.com/craffel/pretty-midi) which are the tree main python libraries parsing MIDI files at the note level.

## Configuration

**Hardware:** Apple M1 Pro cpu, 16GB of memory, macOS 14.4.1

* symusic version: 0.4.5
* miditoolkit version: 1.0.1
* pretty_midi version: 0.2.10

## Results

| Library     | Maestro         | MetaMIDI        | POP909          |
|:------------|:----------------|:----------------|:----------------|
| Symusic     | 1.06 ± 0.89 ms  | 0.37 ± 0.32 ms  | 0.20 ± 0.05 ms  |
| MidiToolkit | 0.11 ± 0.10 sec | 0.04 ± 0.04 sec | 0.02 ± 0.01 sec |
| Pretty MIDI | 0.11 ± 0.10 sec | 0.04 ± 0.04 sec | 0.02 ± 0.01 sec |

miditoolkit and pretty_midi perform equally in average. The two libraries are very similar and both rely on [mido](https://github.com/mido/mido) to read and write MIDI messages.
symusic on the other hand is respectively 104, 108 and 100 times faster than the two others on the Maestro, MetaMIDI and POP909 datasets.
