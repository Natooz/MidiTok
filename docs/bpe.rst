========================
Byte Pair Encoding (BPE)
========================

What is BPE
------------------------

`BPE <https://www.derczynski.com/papers/archive/BPE_Gage.pdf>`_ is a compression technique that replaces the most recurrent byte (tokens in our case) successions of a corpus, by newly created ones.
The most recurrent token successions can be replaced with new created tokens, thus decreasing the sequence length and increasing the vocabulary size.
Today in the NLP field, BPE is used with almost all tokenizations to build their vocabulary, as `it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units <https://aclanthology.org/P16-1162/>`_.
In the case of symbolic, `it has been showned to improve the performances of Transformers models while helping them to learn more isotropic embedding representations <https://arxiv.org/abs/2301.11975>`_.

You can apply it to symbolic music with MidiTok, by first learning the vocabulary (tokenizer.learn_bpe()), and then convert a dataset with BPE (tokenizer.apply_bpe_to_dataset()). All tokenizations not based on embedding pooling are compatible!


Methods
------------------------

.. autofunction:: miditok.MIDITokenizer.learn_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe_to_dataset
    :noindex:

.. autofunction:: miditok.MIDITokenizer.decompose_bpe
    :noindex:

**Tokenizers can be saved and loaded** (:ref:`Save / Load tokenizer`).
After learning BPE (:py:func:`miditok.MIDITokenizer.learn_bpe`), the tokenizer will automatically be saved.
