from .remi import REMI
from .structured import Structured
from .midi_like import MIDILike
from .tsd import TSD
from .cp_word import CPWord
from .mumidi import MuMIDI
from .octuple import Octuple
from .octuple_mono import OctupleMono
from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event

from .utils import utils
from .data_augmentation import data_augmentation


def _changed_class_warning(class_obj):
    print(
        f"\033[93mmiditok warning: {class_obj.__class__.__name__} class has been renamed "
        f"{class_obj.__class__.__bases__[0].__name__} and will be removed in future updates, "
        f"please consider changing it in your code.\033[0m"
    )


class REMIEncoding(REMI):
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
