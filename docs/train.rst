========================
Training a tokenizer
========================

A freshly created tokenizer can already serialize MIDI or abc files into token sequences that can be used to train your model. But if you want to get the best performances (results quality) and efficiency (training and inference speed), **you will need to train the tokenizer first**!

A just created tokenizer will have a vocabulary containing basic tokens representing single attributes of notes, pedals, tempos etc. **Training a tokenizer consists in populating the vocabulary with new tokens representing successions of these basic tokens**, that will be fetched from a training corpus.


Why training
------------------------

If you serialize music files only with these basic tokens, you will encounter two major limitations: your model will not learn meaningful embeddings, and your token sequences will be very large thus hurting the model's efficiency (training/inference speed).

For symbolic music, `training the tokenizer allows to increase both the model's performances and efficiency <https://aclanthology.org/2023.emnlp-main.123/>`_.

Meaningful embeddings
~~~~~~~~~~~~~~~~~~~~~~

During their training, sequential/language models such as Transformers (typically used with MidiTok) learn abstract representations of the tokens, called `embeddings <https://en.wikipedia.org/wiki/Word_embedding>`_, that are vectors in a space with a large number of dimensions (e.g. from 500, up to 10k for the largest models). They do so contextually, depending on how the tokens are present and combined together in the data. This allows them to learn the semantic of the tokens, that can in turn allow them to perform the tasks they are trained for. In other words, they learn the meaning of the words (associated to individual tokens in the vocabulary) to be able to perform their tasks.

In the case of music, newly learned tokens can represent whole notes (i.e. succession of their token attributes) or successions of notes. The notion of semantic is unclear, yet, these embeddings carry more information about the melody and harmony, that the model can learn and leverage.

Reduced sequence lengths
~~~~~~~~~~~~~~~~~~~~~~~~

Serializing music files in single "basic" attribute tokens naturally induces fairly long token sequences. As a note is made of at least three tokens (`Pitch`, `Velocity`, `Duration`/`NoteOff`, optionally `Program`), the resulting token sequence will have a number of tokens at least three times the number of notes.

This is problematic as the time and space complexity of Transformer models grow quadratically with the input sequence length. Thus, the longer the sequence is, the more computations will be made and memory will be used.

Training a tokenizer to learn new tokens that represent combinations of basic tokens will "compress" the sequence, and allow to drastically reduce its number of tokens, and in turn the efficiency of the model it is fed to.


Basic and learned tokens
------------------------

In the case of symbolic music we will consider the tokens of a tokenizer without BPE as the **base vocabulary**, which can be seen as the equivalent of the characters (or bytes) for text. To compute BPE, MidiTok is backed by the Hugging Face `🤗tokenizers <https://github.com/huggingface/tokenizers>`_ Rust library allowing super-fast training and encoding. Thus, internally we represent base tokens (from base vocab) as characters (bytes). Essentially, a token will have three unique (non-shared by others) forms:

* The text describing the token itself, e.g. ``Pitch_58``, ``Position_4`` ...;
* An id as an integer, that will be fed to the model, e.g. ``65``;
* A byte form as a character or succession of characters, e.g. ``a`` or any `unicode character <https://en.wikipedia.org/wiki/List_of_Unicode_characters>`_ starting from the 33rd one (0x21).

A token learned with BPE will be represented by the succession of the unique characters of the base tokens it represent. You can access to several vocabularies to get the equivalents forms of tokens:

* ``vocab``: the base vocabulary, binding token descriptions to their ids;
* ``vocab_bpe``: the vocabulary with BPE applied, binding byte forms to their integer id;
* ``_vocab_base``: a copy of the initial base vocabulary, this attribute is used in case the initial base vocab is overriden by :py:func:`miditok.MIDITokenizer.train` with the ``start_from_empty_voc`` option;
* ``_vocab_base``:
* ``_vocab_base_byte_to_token``: biding the base token byte forms to their string forms;
* ``_vocab_base_id_to_byte``: biding the base token ids (integers) to their byte forms;
* ``_vocab_bpe_bytes_to_tokens``: biding the byte forms of the complete vocab to their string forms, as a list of string;

For symbolic music, BPE `has been showned to improve the performances of Transformers models while helping them to learn more isotropic embedding representations <https://aclanthology.org/2023.emnlp-main.123/>`_. BPE can be applied on top of any tokenization, as long as it is not based on embedding pooling! (``is_multi_voc``)



Tokenizer models
------------------------

Byte Pair Encoding (BPE)
~~~~~~~~~~~~~~~~~~~~~~~~

`BPE <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_ is a compression technique that replaces the most recurrent byte (tokens in our case) successions of a corpus, by newly created ones.
For instance, in the character sequence ``aabaabaacaa``, the sub-sequence ``aa`` occurs three times and is the most recurrent one. Learning and applying BPE on this sequence would replace ``aa`` with a new symbol, e.g., `d`, resulting in a compressed sequence ``dbdbdcd``. The latter can be reduced again by replacing the ``db`` subsequence, giving ``eedcd``. The vocabulary, which initially contained three characters (``a``, ``b`` and ``c``) now also contains ``d`` and ``e``. In practice BPE is learned on a corpus until the vocabulary reaches a target size.

Today in the NLP field, BPE is used with almost all tokenizations to build their vocabulary, as `it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units <https://aclanthology.org/P16-1162/>`_. The base initial vocabulary is the set of all the unique characters present in the data, which compose the words that are automatically learned as tokens by the BPE algorithm.

Unigram
~~~~~~~~~~~~~~~~~~~~~~~~

WordPiece
~~~~~~~~~~~~~~~~~~~~~~~~


Splitting the ids
------------------------



Code example
------------------------

..  code-block:: python

    from miditok import REMI, TokenizerConfig, TokSequence
    from copy import deepcopy

    tokenizer = REMI(TokenizerConfig(use_programs=True))
    paths_midis = list(Path("path", "to", "midis").glob('**/*.mid'))

    # Learns the vocabulary with BPE
    # Ids are split per bars by default
    tokenizer.train(
        vocab_size=500,
        model="BPE",
        files_paths=paths_midis,
    )

    # Tokenize a MIDI file
    tokens = tokenizer(paths_midis[0])
    # Decode BPE
    tokens_no_bpe = tokenizer.decode_bpe(deepcopy(tok_seq))

Methods
------------------------

A tokenizer can be trained with the :py:func:`miditok.MIDITokenizer.train` method. After being trained, the tokenizer will automatically encode the token ids with its model when tokenizing music files.

Trained tokenizers can be saved and loaded back (:ref:`Save / Load tokenizer`).

.. autofunction:: miditok.MIDITokenizer.train
    :noindex:

.. autofunction:: miditok.MIDITokenizer.encode_ids
    :noindex:

.. autofunction:: miditok.MIDITokenizer.decode_ids
    :noindex:
