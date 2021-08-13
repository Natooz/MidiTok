""" Constants for data encoding

"""

# MIDI encodings default parameters, used when tokenizing a dataset and using tokens
# These are the parameters from which a MIDI file will be tokenized
PITCH_RANGE = range(21, 109)  # the recommended pitches for piano in the GM2 specs are from 21 to 108
BEAT_RES = {(0, 4): 8, (4, 12): 4}  # samples per beat
NB_VELOCITIES = 32  # nb of velocity bins, velocities values from 0 to 127 will be quantized
ADDITIONAL_TOKENS = {'Chord': True,
                     'Empty': True,
                     'Tempo': False,  # Unused for now (not implemented)
                     'Ignore': True}  # for CP words only

# Used when creating the event <--> token dictionary
PROGRAM_TOKENS = True  # will include tokens specifying the instrument of each sequence at its beginning

# Defaults when writing new MIDI files, 384 and 480 are convenient as divisible by 4, 8, 12, 16, 24, 32
TIME_DIVISION = 384

CHORD_MAPS = {'min': (0, 3, 7),
              'maj': (0, 4, 7),
              'dim': (0, 3, 6),
              'aug': (0, 4, 8),
              'sus2': (0, 2, 7),
              'sus4': (0, 5, 7),

              '7dom': (0, 4, 7, 10),
              '7min': (0, 3, 7, 10),
              '7maj': (0, 4, 7, 11),
              '7halfdim': (0, 3, 6, 10),
              '7dim': (0, 3, 6, 9),
              '7aug': (0, 4, 8, 11),

              '9maj': (0, 4, 7, 10, 14),
              '9min': (0, 4, 7, 10, 13)}


# http://newt.phys.unsw.edu.au/jw/notes.html
# https://www.midi.org/specifications

# index i = program i+1 in the GM2 specs (7. Appendix A)
# index i = program i as retrieved by packages like mido or miditoolkit
MIDI_INSTRUMENTS = [{'name': 'Acoustic Grand Piano', 'pitch_range': (21, 108)},
                    {'name': 'Bright Acoustic Piano', 'pitch_range': (21, 108)},
                    {'name': 'Electric Grand Piano', 'pitch_range': (21, 108)},
                    {'name': 'Honky-tonk Piano', 'pitch_range': (21, 108)},
                    {'name': 'Electric Piano 1', 'pitch_range': (28, 103)},
                    {'name': 'Electric Piano 2', 'pitch_range': (28, 103)},
                    {'name': 'Harpsichord', 'pitch_range': (41, 89)},
                    {'name': 'Clavi', 'pitch_range': (36, 96)},

                    # Chromatic Percussion
                    {'name': 'Celesta', 'pitch_range': (60, 108)},
                    {'name': 'Glockenspiel', 'pitch_range': (72, 108)},
                    {'name': 'Music Box', 'pitch_range': (60, 84)},
                    {'name': 'Vibraphone', 'pitch_range': (53, 89)},
                    {'name': 'Marimba', 'pitch_range': (48, 84)},
                    {'name': 'Xylophone', 'pitch_range': (65, 96)},
                    {'name': 'Tubular Bells', 'pitch_range': (60, 77)},
                    {'name': 'Dulcimer', 'pitch_range': (60, 84)},

                    # Organs
                    {'name': 'Drawbar Organ', 'pitch_range': (36, 96)},
                    {'name': 'Percussive Organ', 'pitch_range': (36, 96)},
                    {'name': 'Rock Organ', 'pitch_range': (36, 96)},
                    {'name': 'Church Organ', 'pitch_range': (21, 108)},
                    {'name': 'Reed Organ', 'pitch_range': (36, 96)},
                    {'name': 'Accordion', 'pitch_range': (53, 89)},
                    {'name': 'Harmonica', 'pitch_range': (60, 84)},
                    {'name': 'Tango Accordion', 'pitch_range': (53, 89)},

                    # Guitars
                    {'name': 'Acoustic Guitar (nylon)', 'pitch_range': (40, 84)},
                    {'name': 'Acoustic Guitar (steel)', 'pitch_range': (40, 84)},
                    {'name': 'Electric Guitar (jazz)', 'pitch_range': (40, 86)},
                    {'name': 'Electric Guitar (clean)', 'pitch_range': (40, 86)},
                    {'name': 'Electric Guitar (muted)', 'pitch_range': (40, 86)},
                    {'name': 'Overdriven Guitar', 'pitch_range': (40, 86)},
                    {'name': 'Distortion Guitar', 'pitch_range': (40, 86)},
                    {'name': 'Guitar Harmonics', 'pitch_range': (40, 86)},

                    # Bass
                    {'name': 'Acoustic Bass', 'pitch_range': (28, 55)},
                    {'name': 'Electric Bass (finger)', 'pitch_range': (28, 55)},
                    {'name': 'Electric Bass (pick)', 'pitch_range': (28, 55)},
                    {'name': 'Fretless Bass', 'pitch_range': (28, 55)},
                    {'name': 'Slap Bass 1', 'pitch_range': (28, 55)},
                    {'name': 'Slap Bass 2', 'pitch_range': (28, 55)},
                    {'name': 'Synth Bass 1', 'pitch_range': (28, 55)},
                    {'name': 'Synth Bass 2', 'pitch_range': (28, 55)},

                    # Strings & Orchestral instruments
                    {'name': 'Violin', 'pitch_range': (55, 93)},
                    {'name': 'Viola', 'pitch_range': (48, 84)},
                    {'name': 'Cello', 'pitch_range': (36, 72)},
                    {'name': 'Contrabass', 'pitch_range': (28, 55)},
                    {'name': 'Tremolo Strings', 'pitch_range': (28, 93)},
                    {'name': 'Pizzicato Strings', 'pitch_range': (28, 93)},
                    {'name': 'Orchestral Harp', 'pitch_range': (23, 103)},
                    {'name': 'Timpani', 'pitch_range': (36, 57)},

                    # Ensembles
                    {'name': 'String Ensembles 1', 'pitch_range': (28, 96)},
                    {'name': 'String Ensembles 2', 'pitch_range': (28, 96)},
                    {'name': 'SynthStrings 1', 'pitch_range': (36, 96)},
                    {'name': 'SynthStrings 2', 'pitch_range': (36, 96)},
                    {'name': 'Choir Aahs', 'pitch_range': (48, 79)},
                    {'name': 'Voice Oohs', 'pitch_range': (48, 79)},
                    {'name': 'Synth Voice', 'pitch_range': (48, 84)},
                    {'name': 'Orchestra Hit', 'pitch_range': (48, 72)},

                    # Brass
                    {'name': 'Trumpet', 'pitch_range': (58, 94)},
                    {'name': 'Trombone', 'pitch_range': (34, 75)},
                    {'name': 'Tuba', 'pitch_range': (29, 55)},
                    {'name': 'Muted Trumpet', 'pitch_range': (58, 82)},
                    {'name': 'French Horn', 'pitch_range': (41, 77)},
                    {'name': 'Brass Section', 'pitch_range': (36, 96)},
                    {'name': 'Synth Brass 1', 'pitch_range': (36, 96)},
                    {'name': 'Synth Brass 2', 'pitch_range': (36, 96)},

                    # Reed
                    {'name': 'Soprano Sax', 'pitch_range': (54, 87)},
                    {'name': 'Alto Sax', 'pitch_range': (49, 80)},
                    {'name': 'Tenor Sax', 'pitch_range': (42, 75)},
                    {'name': 'Baritone Sax', 'pitch_range': (37, 68)},
                    {'name': 'Oboe', 'pitch_range': (58, 91)},
                    {'name': 'English Horn', 'pitch_range': (52, 81)},
                    {'name': 'Bassoon', 'pitch_range': (34, 72)},
                    {'name': 'Clarinet', 'pitch_range': (50, 91)},

                    # Pipe
                    {'name': 'Piccolo', 'pitch_range': (74, 108)},
                    {'name': 'Flute', 'pitch_range': (60, 96)},
                    {'name': 'Recorder', 'pitch_range': (60, 96)},
                    {'name': 'Pan Flute', 'pitch_range': (60, 96)},
                    {'name': 'Blown Bottle', 'pitch_range': (60, 96)},
                    {'name': 'Shakuhachi', 'pitch_range': (55, 84)},
                    {'name': 'Whistle', 'pitch_range': (60, 96)},
                    {'name': 'Ocarina', 'pitch_range': (60, 84)},

                    # Synth Lead
                    {'name': 'Lead 1 (square)', 'pitch_range': (21, 108)},
                    {'name': 'Lead 2 (sawtooth)', 'pitch_range': (21, 108)},
                    {'name': 'Lead 3 (calliope)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 4 (chiff)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 5 (charang)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 6 (voice)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 7 (fifths)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 8 (bass + lead)', 'pitch_range': (21, 108)},

                    # Synth Pad
                    {'name': 'Pad 1 (new age)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 2 (warm)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 3 (polysynth)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 4 (choir)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 5 (bowed)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 6 (metallic)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 7 (halo)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 8 (sweep)', 'pitch_range': (36, 96)},

                    # Synth SFX
                    {'name': 'FX 1 (rain)', 'pitch_range': (36, 96)},
                    {'name': 'FX 2 (soundtrack)', 'pitch_range': (36, 96)},
                    {'name': 'FX 3 (crystal)', 'pitch_range': (36, 96)},
                    {'name': 'FX 4 (atmosphere)', 'pitch_range': (36, 96)},
                    {'name': 'FX 5 (brightness)', 'pitch_range': (36, 96)},
                    {'name': 'FX 6 (goblins)', 'pitch_range': (36, 96)},
                    {'name': 'FX 7 (echoes)', 'pitch_range': (36, 96)},
                    {'name': 'FX 8 (sci-fi)', 'pitch_range': (36, 96)},

                    # Ethnic Misc.
                    {'name': 'Sitar', 'pitch_range': (48, 77)},
                    {'name': 'Banjo', 'pitch_range': (48, 84)},
                    {'name': 'Shamisen', 'pitch_range': (50, 79)},
                    {'name': 'Koto', 'pitch_range': (55, 84)},
                    {'name': 'Kalimba', 'pitch_range': (48, 79)},
                    {'name': 'Bag pipe', 'pitch_range': (36, 77)},
                    {'name': 'Fiddle', 'pitch_range': (55, 96)},
                    {'name': 'Shanai', 'pitch_range': (48, 72)},

                    # Percussive
                    {'name': 'Tinkle Bell', 'pitch_range': (72, 84)},
                    {'name': 'Agogo', 'pitch_range': (60, 72)},
                    {'name': 'Steel Drums', 'pitch_range': (52, 76)},
                    {'name': 'Woodblock', 'pitch_range': (0, 127)},
                    {'name': 'Taiko Drum', 'pitch_range': (0, 127)},
                    {'name': 'Melodic Tom', 'pitch_range': (0, 127)},
                    {'name': 'Synth Drum', 'pitch_range': (0, 127)},
                    {'name': 'Reverse Cymbal', 'pitch_range': (0, 127)},

                    # SFX
                    {'name': 'Guitar Fret Noise, Guitar Cutting Noise', 'pitch_range': (0, 127)},
                    {'name': 'Breath Noise, Flute Key Click', 'pitch_range': (0, 127)},
                    {'name': 'Seashore, Rain, Thunder, Wind, Stream, Bubbles', 'pitch_range': (0, 127)},
                    {'name': 'Bird Tweet, Dog, Horse Gallop', 'pitch_range': (0, 127)},
                    {'name': 'Telephone Ring, Door Creaking, Door, Scratch, Wind Chime', 'pitch_range': (0, 127)},
                    {'name': 'Helicopter, Car Sounds', 'pitch_range': (0, 127)},
                    {'name': 'Applause, Laughing, Screaming, Punch, Heart Beat, Footstep', 'pitch_range': (0, 127)},
                    {'name': 'Gunshot, Machine Gun, Lasergun, Explosion', 'pitch_range': (0, 127)}]

INSTRUMENT_CLASSES = dict([(n, (0, 'Piano')) for n in range(0, 8)] +
                          [(n, (1, 'Chromatic Percussion')) for n in range(8, 16)] +
                          [(n, (2, 'Organ')) for n in range(16, 24)] +
                          [(n, (3, 'Guitar')) for n in range(24, 32)] +
                          [(n, (4, 'Bass')) for n in range(32, 40)] +
                          [(n, (5, 'Strings')) for n in range(40, 48)] +
                          [(n, (6, 'Ensemble')) for n in range(48, 56)] +
                          [(n, (7, 'Brass')) for n in range(56, 64)] +
                          [(n, (8, 'Reed')) for n in range(64, 72)] +
                          [(n, (9, 'Pipe')) for n in range(72, 80)] +
                          [(n, (10, 'Synth Lead')) for n in range(80, 88)] +
                          [(n, (11, 'Synth Pad')) for n in range(88, 96)] +
                          [(n, (12, 'Synth Effects')) for n in range(96, 104)] +
                          [(n, (13, 'Ethnic')) for n in range(104, 112)] +
                          [(n, (14, 'Percussive')) for n in range(112, 120)] +
                          [(n, (15, 'Sound Effects')) for n in range(120, 128)] +
                          [(-1, (-1, 'Drums'))])

INSTRUMENT_CLASSES_RANGES = {'Piano': (0, 7), 'Chromatic Percussion': (8, 15), 'Organ': (16, 23), 'Guitar': (24, 31),
                             'Bass': (32, 39), 'Strings': (40, 47), 'Ensemble': (48, 55), 'Brass': (56, 63),
                             'Reed': (64, 71),
                             'Pipe': (72, 79), 'Synth Lead': (80, 87), 'Synth Pad': (88, 95),
                             'Synth Effects': (96, 103),
                             'Ethnic': (104, 111), 'Percussive': (112, 119), 'Sound Effects': (120, 127), 'Drums': -1}

# index i = program i+1 in the GM2 specs (8. Appendix B)
# index i = program i as retrieved by packages like mido or miditoolkit
DRUM_SETS = {0: 'Standard', 8: 'Room', 16: 'Power', 24: 'Electronic', 25: 'Analog', 32: 'Jazz', 40: 'Brush',
             48: 'Orchestra', 56: 'SFX'}
