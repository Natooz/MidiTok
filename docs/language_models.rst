===================================
Language models 101
===================================

This page introduces the basic concepts of music, the MIDI protocol and sequential deep learning models. It aims to bring the basic knowledge around this subjects in order to understand how to use music with AI models, without going into too specific details, for which more comprehensive references are attached.




Token, vocabulary, token id, embedding

Sequential / Language models
----------------------------

A token is a distinct element, part of a sequence of tokens. In natural language, a token can be a character, a subword or a word. A sentence can then be tokenized into a sequence of tokens representing the words and punctuation.
For symbolic music, tokens can represent the values of the note attributes (pitch, valocity, duration) or time events. These are the "basic" tokens, that can be compared to the characters in natural language. In the vocabulary of trained tokenizers, the tokens can represent **successions** of these basic tokens.
A token can take three forms, which we name by convention:

* Token (``string``): the form describing it, e.g. *Pitch_50*.
* Id (``int``): an unique associated integer, which corresponds to the index of the index in the vocabulary.
* Byte (``string``): an distinct byte, used internally for trained tokenizers (:ref:`Training a tokenizer`).

MidiTok works with :ref:`TokSequence` objects to conveniently represent these three forms.


The vocabulary of a tokenizer acts as a lookup table, linking tokens (string / byte) to their ids (integer). The vocabulary is an attribute of the tokenizer and can be accessed with ``tokenizer.vocab``. The vocabulary is a Python dictionary binding tokens (keys) to their ids (values).
For tokenizations with embedding pooling (e.g. :ref:`CPWord` or :ref:`Octuple`), ``tokenizer.vocab`` will be a list of ``Vocabulary`` objects, and the ``tokenizer.is_multi_vocab`` property will be ``True``.

**With Byte Pair Encoding:**
``tokenizer.vocab`` holds all the basic tokens describing the note and time attributes of music. By analogy with text, these tokens can be seen as unique characters.
After :ref:`Training a tokenizer`, a new vocabulary is built with newly created tokens from pairs of basic tokens. This vocabulary can be accessed with ``tokenizer.vocab_bpe``, and binds tokens as bytes (string) to their associated ids (int). This is the vocabulary of the 🤗tokenizers BPE model.
