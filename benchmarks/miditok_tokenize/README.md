# Tokenization times

This benchmark measures the tokenization times of MIDI files from the [Maestro](https://magenta.tensorflow.org/datasets/maestro), [Lakh](https://colinraffel.com/projects/lmd/) and [POP909](https://arxiv.org/abs/2008.07142) datasets.

## Configuration

**Hardware:** Apple M1 Pro cpu, 16GB of memory, macOS 14.4.1

* miditok: v3.0.3
* symusic: v0.4.5
* tokenizers: v0.19.0
* numpy: v1.26.4

* Maximum number of files per dataset for analysis: 1k
* Using tempo, time signature, sustain pedal and pitch bend tokens

## Results

|            | Maestro        | MMD            | POP909        |
|:-----------|:---------------|:---------------|:--------------|
| REMI       | 38.97±32.92 ms | 24.55±52.25 ms | 11.00±7.73 ms |
| TSD        | 52.62±41.59 ms | 31.70±73.93 ms | 13.35±7.66 ms |
| MIDILike   | 61.75±48.27 ms | 36.28±76.87 ms | 17.77±8.91 ms |
| Structured | 60.38±46.78 ms | 35.85±88.48 ms | 16.56±8.62 ms |
