# Tokenizer training benchmark

This benchmark aims to measure the training speeds of the different tokenization algorithms, as well as their encoding-decoding speeds, sequence length reduction, and the impact of some other strategies such as spitting the tokens per bars or beats.

## Configuration

### Hardware

Apple M1 Pro, 16GB of memory, macOS 14.4.1

### Software

* miditok: v3.0.3
* symusic: v0.4.5
* tokenizers: v0.19.0
* numpy: v1.26.4

### Parameters

* Maximum number of files per dataset for training: 20k
* Maximum number of files per dataset for analysis: 2k
* Using tempo, time signature, rests, sustain pedal and pitch bend tokens

## Training times

## Splitting ids per bars and beats

Measures the sequence length of subsequence obtained when splitting the token sequences of whole music files per bars or beats.

|                 | Maestro           | Lakh                 |  Lakh monotrack   |
|:----------------|:------------------|:---------------------|:------------------|
| REMI - bar      | 74.7±45.8 (↑ 460) | 107.1±129.6 (↑ 2525) | 12.5±24.1 (↑ 624) |
| REMI - beat     | 18.7±13.1 (↑ 190) | 27.4±34.5 (↑ 659)    | 3.3±6.6 (↑ 307)   |
| TSD - bar       | 70.9±44.3 (↑ 456) | 105.7±128.8 (↑ 2521) | 11.2±22.3 (↑ 623) |
| TSD - beat      | 17.7±12.7 (↑ 188) | 27.1±34.2 (↑ 658)    | 2.9±6.1 (↑ 306)   |
| MIDILike - bar  | 77.5±45.9 (↑ 461) | 133.7±163.5 (↑ 3154) | 11.7±23.8 (↑ 624) |
| MIDILike - beat | 19.4±12.8 (↑ 183) | 34.2±43.1 (↑ 832)    | 3.1±6.5 (↑ 317)   |

Main observation: beat subsequences are relatively shorts, and in average four times larger than bar sequences, as most files have 4/* time signatures.

## WordPiece `max_input_chars_per_word` impact

Analyze the impact of the `max_input_chars_per_word` parameter of the WordPiece model, on training and encoding times.
The vocabulary size used here is 20k.

### Training time

|      | Maestro no-split   | Maestro bar-split   | Maestro beat-split   | Lakh multitrack no-split   | Lakh multitrack bar-split   | Lakh multitrack beat-split   |
|-----:|:-------------------|:--------------------|:---------------------|:---------------------------|:----------------------------|:-----------------------------|
|  100 | 131.9 sec          | 88.2 sec            | 99.3 sec             | 1216.5 sec                 | 1463.9 sec                  | 1538.3 sec                   |
|  200 | 128.4 sec          | 88.2 sec            | 98.2 sec             | 1140.3 sec                 | 1283.4 sec                  | 1505.6 sec                   |
|  500 | 128.1 sec          | 86.6 sec            | 98.2 sec             | 1171.8 sec                 | 1457.4 sec                  | 1604.2 sec                   |
| 1000 | 127.8 sec          | 86.4 sec            | 97.0 sec             | 1131.1 sec                 | 1390.0 sec                  | 1620.8 sec                   |
| 2000 | 128.5 sec          | 86.0 sec            | 96.7 sec             | 1238.1 sec                 | 1431.2 sec                  | 1495.7 sec                   |
| 5000 | 127.1 sec          | 85.5 sec            | 96.7 sec             | 1229.0 sec                 | 1543.7 sec                  | 1709.8 sec                   |

`max_input_chars_per_word` has almost no impact on the training time.

### Encoding time and ratio of "unknown token"

|      | Maestro no-split          | Maestro bar-split         | Maestro beat-split        | Lakh multitrack no-split   | Lakh multitrack bar-split   | Lakh multitrack beat-split   |
|-----:|:--------------------------|:--------------------------|:--------------------------|:---------------------------|:----------------------------|:-----------------------------|
|  100 | 0.0030±0.0022 (1.000 unk) | 0.0195±0.0156 (0.001 unk) | 0.0238±0.0200 (0.000 unk) | 0.0003±0.0004 (0.937 unk)  | 0.0026±0.0159 (0.007 unk)   | 0.0044±0.0495 (0.007 unk)    |
|  200 | 0.0030±0.0022 (1.000 unk) | 0.0416±0.0332 (0.000 unk) | 0.0239±0.0199 (0.000 unk) | 0.0004±0.0005 (0.866 unk)  | 0.0027±0.0146 (0.007 unk)   | 0.0038±0.0475 (0.007 unk)    |
|  500 | 0.0029±0.0022 (1.000 unk) | 0.0443±0.0365 (0.000 unk) | 0.0235±0.0197 (0.000 unk) | 0.0010±0.0016 (0.698 unk)  | 0.0029±0.0156 (0.007 unk)   | 0.0038±0.0466 (0.007 unk)    |
| 1000 | 0.0030±0.0022 (0.999 unk) | 0.0442±0.0366 (0.000 unk) | 0.0236±0.0202 (0.000 unk) | 0.0057±0.0115 (0.513 unk)  | 0.0032±0.0165 (0.007 unk)   | 0.0039±0.0478 (0.007 unk)    |
| 2000 | 0.0037±0.0127 (0.996 unk) | 0.0442±0.0364 (0.000 unk) | 0.0232±0.0194 (0.000 unk) | 0.0405±0.0771 (0.301 unk)  | 0.0029±0.0159 (0.007 unk)   | 0.0042±0.0475 (0.007 unk)    |
| 5000 | 0.1209±0.6198 (0.955 unk) | 0.0440±0.0363 (0.000 unk) | 0.0238±0.0208 (0.000 unk) | 0.3539±0.8183 (0.102 unk)  | 0.0034±0.0174 (0.007 unk)   | 0.0043±0.0501 (0.007 unk)    |

`max_input_chars_per_word` has however a significant negative impact on the encoding time of the token ids.
The ratios of unknown tokens also highlight the **importance of splitting the token ids per bars or beats**. Not doing so results in either a high proportion of unknown tokens with low `max_input_chars_per_word` values thus loosing data integrity, or with very high encoding times for high `max_input_chars_per_word` values.
