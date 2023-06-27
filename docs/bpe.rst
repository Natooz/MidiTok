========================
Byte Pair Encoding (BPE)
========================

What is BPE
------------------------

`BPE <https://www.derczynski.com/papers/archive/BPE_Gage.pdf>`_ is a compression technique that replaces the most recurrent byte (tokens in our case) successions of a corpus, by newly created ones.
For instance, in the character sequence ``aabaabaacaa``, the sub-sequence ``aa`` occurs three times and is the most recurrent one. Learning and applying BPE on this sequence would replace ``aa`` with a new symbol, e.g., `d`, resulting in a compressed sequence ``dbdbdcd``. The latter can be reduced again by replacing the ``db`` subsequence, giving ``eedcd``. The vocabulary, which initially contained three characters (``a``, ``b`` and ``c``) now also contains ``d`` and ``e``. In practice BPE is learned on a corpus until the vocabulary reaches a target size.

Today in the NLP field, BPE is used with almost all tokenizations to build their vocabulary, as `it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units <https://aclanthology.org/P16-1162/>`_. The base initial vocabulary is the set of all the unique characters present in the data, which compose the words that are automatically learned as tokens by the BPE algorithm.

BPE for symbolic music
------------------------

In the case of symbolic music we will consider the tokens of a tokenizer without BPE as the **base vocabulary**, which can be seen as the equivalent of the characters (or bytes) for text. To compute BPE, MidiTok is backed by the Hugging Face `🤗tokenizers <https://github.com/huggingface/tokenizers>`_ Rust library allowing super-fast training and encoding. Thus, internally we represent base tokens (from base vocab) as characters (bytes). Essentially, a token will have three unique (non-shared by others) forms:

* The text describing the token itself, e.g. ``Pitch_58``, ``Position_4`` ...;
* An id as an integer, that will be fed to the model, e.g. ``65``;
* A byte form as a character or succession of characters, e.g. ``a`` or any `unicode character <https://en.wikipedia.org/wiki/List_of_Unicode_characters>`_ starting from the 33rd one (0x21).

A token learned with BPE will be represented by the succession of the unique characters of the base tokens it represent. You can access to several vocabularies to get the equivalents forms of tokens:

* ``vocab``: the base vocabulary, binding token descriptions to their ids;
* ``vocab_bpe``: the vocabulary with BPE applied, binding byte forms to their integer id;
* ``_vocab_base``: a copy of the initial base vocabulary, this attribute is used in case the initial base vocab is overriden by :py:func:`miditok.MIDITokenizer.learn_bpe` with the ``start_from_empty_voc`` option;
* ``_vocab_base``:
* ``_vocab_base_byte_to_token``: biding the base token byte forms to their string forms;
* ``_vocab_base_id_to_byte``: biding the base token ids (integers) to their byte forms;
* ``_vocab_bpe_bytes_to_tokens``: biding the byte forms of the complete vocab to their string forms, as a list of string;

For symbolic music, BPE `has been showned to improve the performances of Transformers models while helping them to learn more isotropic embedding representations <https://arxiv.org/abs/2301.11975>`_. BPE can be applied on top of any tokenization, as long as it is not based on embedding pooling! (``is_multi_voc``)

BPE example
------------------------

..  code-block:: python

    from miditok import REMI, TokSequence
    from copy import deepcopy

    tokenizer = REMI()  # using defaults parameters (constants.py)
    token_paths = list(Path('path', 'to', 'dataset').glob('**/*.json'))

    # Learns the vocabulary with BPE
    tokenizer.learn_bpe(
        vocab_size=500,
        tokens_paths=list(Path('path', 'to', 'tokens_noBPE').glob("**/*.json")),
        out_dir=Path('path', 'to', 'tokens_BPE'),
    )

    # Opens tokens, apply BPE on them, and decode BPE back
    tokens = tokenizer.load_tokens(token_paths[0])
    tokens = TokSequence(ids=tokens)
    tokens_with_bpe = tokenizer.apply_bpe(deepcopy(tokens))  # copy as the method is inplace
    tokens_no_bpe = tokenizer.decode_bpe(deepcopy(tokens_with_bpe))

    # Converts the tokenized musics into tokens with BPE
    tokenizer.apply_bpe_to_dataset(Path('path', 'to', 'tokens_noBPE'), Path('path', 'to', 'tokens_BPE'))


Methods
------------------------

To use BPE, you must first train your tokenizer from data (:py:func:`miditok.MIDITokenizer.learn_bpe`), and then convert a dataset with BPE (:py:func:`miditok.MIDITokenizer.apply_bpe_to_dataset`).

**Tokenizers can be saved and loaded** (:ref:`Save / Load tokenizer`).

.. autofunction:: miditok.MIDITokenizer.learn_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe
    :noindex:

.. autofunction:: miditok.MIDITokenizer.apply_bpe_to_dataset
    :noindex:

.. autofunction:: miditok.MIDITokenizer.decode_bpe
    :noindex:
