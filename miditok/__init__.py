from .remi import REMI
from .structured import Structured
from .midi_like import MIDILike
from .cp_word import CPWord
from .mumidi import MuMIDI
from .octuple import Octuple
from .octuple_mono import OctupleMono
from .midi_tokenizer_base import MIDITokenizer, get_midi_programs, detect_chords, merge_tracks, \
    merge_same_program_tracks
from .vocabulary import Vocabulary, Event


def _changed_class_warning(class_obj):
    print(f'\033[93mmiditok: {class_obj.__class__.__name__} class has been renamed '
          f'{class_obj.__class__.__bases__[0].__name__} and will be removed in future updates, '
          f'please consider changing it in your code\033[0m')


class REMIEncoding(REMI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class StructuredEndcoding(Structured):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class MIDILikeEncoding(MIDILike):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class CPWordEncoding(CPWord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class MuMIDIEncoding(MuMIDI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class OctupleEncoding(Octuple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)


class OctupleMonoEncoding(OctupleMono):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _changed_class_warning(self)
