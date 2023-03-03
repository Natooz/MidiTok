========================
Byte Pair Encoding (BPE)
========================

What is BPE
------------------------

`BPE <https://www.derczynski.com/papers/archive/BPE_Gage.pdf>`_ is a compression technique that replaces the most recurrent byte (tokens in our case) successions of a corpus, by newly created ones.
The most recurrent token successions can be replaced with new created tokens, thus decreasing the sequence length and increasing the vocabulary size.
Today in the NLP field, BPE is used with almost all tokenizations to build their vocabulary, as `it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units <https://aclanthology.org/P16-1162/>`_.
In the case of symbolic, `it has been showned to improve the performances of Transformers models while helping them to learn more isotropic embedding representations <https://arxiv.org/abs/2301.11975>`_.

MidiTok allows to use BPE for symbolic music, on top of any tokenizations not based on embedding pooling!
BPE is backed by the Hugging Face `🤗tokenizers <https://github.com/huggingface/tokenizers>`_ Rust library for fast training and encoding. You can also use the slow 100% Python alternative, even though it will be about 30 times slower.
To use BPE, you must first train your tokenizer from data (:py:func:`miditok.MIDITokenizer.learn_bpe`), and then convert a dataset with BPE (:py:func:`miditok.MIDITokenizer.apply_bpe_to_dataset`).

**Tokenizers can be saved and loaded** (:ref:`Save / Load tokenizer`).

Methods
------------------------

.. autofunction:: miditok.MIDITokenizer.learn_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe_to_dataset
    :noindex:

.. autofunction:: miditok.MIDITokenizer.decode_bpe
    :noindex:

Slow methods
------------------------

To use the slow Python BPE, just train your tokenizer with :py:func:`miditok.MIDITokenizer.learn_bpe_slow`, and use the methods from the previous section.

.. autofunction:: miditok.MIDITokenizer.learn_bpe_slow
    :noindex:
