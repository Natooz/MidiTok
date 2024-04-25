# MidiTok preprocessing

This benchmark measures the preprocessing times of MIDI files, performed by MidiTok with the `tokenizer.preprocess_score` method.

## Configuration

**Hardware:** Apple M1 Pro cpu, 16GB of memory, macOS 14.4.1

* Maximum number of files per dataset for analysis: 1k
* Using tempo, time signature, sustain pedal and pitch bend tokens

## Results

|               | symusic version   | Maestro - REMI   | Maestro - TSD   | Maestro - MIDILike   | Maestro - Structured   | MMD - REMI   | MMD - TSD    | MMD - MIDILike   | MMD - Structured   | POP909 - REMI   | POP909 - TSD   | POP909 - MIDILike   | POP909 - Structured   |
|:--------------|:------------------|:-----------------|:----------------|:---------------------|:-----------------------|:-------------|:-------------|:-----------------|:-------------------|:----------------|:---------------|:--------------------|:----------------------|
| miditok 3.0.3 | 0.4.5             | 0.64±0.36 ms     | 0.62±0.35 ms    | 0.47±0.25 ms         | 0.46±0.32 ms           | 1.55±3.68 ms | 1.54±3.68 ms | 1.40±3.63 ms     | 0.40±0.51 ms       | 0.32±0.07 ms    | 0.30±0.07 ms   | 0.24±0.06 ms        | 0.16±0.03 ms          |
