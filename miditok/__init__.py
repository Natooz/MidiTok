from .remi import REMIEncoding
from .structured import StructuredEncoding
from .midi_like import MIDILikeEncoding
from .cp_word import CPWordEncoding
from .midi_encoding_base import quantize_note_times, detect_chords, merge_tracks, current_bar_pos, MIDIEncoding
from .constants import MIDI_INSTRUMENTS, INSTRUMENT_CLASSES, INSTRUMENT_CLASSES_RANGES, CHORD_MAPS, DRUM_SETS
