.. _training-tokenizer-label:

========================
Training a tokenizer
========================

A freshly created tokenizer can already serialize MIDI or abc files into token sequences that can be used to train your model. But if you want to get the best performances (results quality) and efficiency (training and inference speed), **you will need to train the tokenizer first**!

A just created tokenizer will have a vocabulary containing basic tokens representing single attributes of notes, pedals, tempos etc. **Training a tokenizer consists in populating the vocabulary with new tokens representing successions of these basic tokens**, that will be fetched from a training corpus.

All tokenizers can be trained, except if they use embedding pooling (``is_multi_voc``)!


Why training a tokenizer
------------------------

If you serialize music files only with these basic tokens, you will encounter two major limitations: your model will not learn meaningful embeddings, and your token sequences will be very large thus hurting the model's efficiency (training/inference speed).

For symbolic music, `training the tokenizer allows to increase both the model's performances and efficiency <https://aclanthology.org/2023.emnlp-main.123/>`_.

Meaningful embeddings
~~~~~~~~~~~~~~~~~~~~~~

During their training, sequential/language models such as Transformers (typically used with MidiTok) learn abstract representations of the tokens, called `embeddings <https://en.wikipedia.org/wiki/Word_embedding>`_, that are vectors in a space with a large number of dimensions (e.g. from 500, up to 10k for the largest models). They do so contextually, depending on how the tokens are present and combined together in the data. This allows them to learn the semantic of the tokens, that can in turn allow them to perform the tasks they are trained for. In other words, they learn the meaning of the words (associated to individual tokens in the vocabulary) to be able to perform their tasks.

In the case of music, newly learned tokens can represent whole notes (i.e. succession of their token attributes) or successions of notes. The notion of semantic is unclear, yet, these embeddings carry more information about the melody and harmony, that the model can learn and leverage.

Reduced sequence lengths
~~~~~~~~~~~~~~~~~~~~~~~~

Serializing music files in single "basic" attribute tokens naturally induces fairly long token sequences. As a note is made of at least three tokens (``Pitch``, ``Velocity``, ``Duration``/``NoteOff``, optionally ``Program``), the resulting token sequence will have a number of tokens at least three times the number of notes.

This is problematic as the time and space complexity of Transformer models grow quadratically with the input sequence length. Thus, the longer the sequence is, the more computations will be made and memory will be used.

Training a tokenizer to learn new tokens that represent combinations of basic tokens will "compress" the sequence, and allow to drastically reduce its number of tokens, and in turn the efficiency of the model it is fed to.


Basic and learned tokens
------------------------

A tokenizer features an **base vocabulary**, which contains the tokens representing each note attributes, tempos, times etc. This base vocabulary is created from the values you set in the tokenizer's config (e.g. list of pitches, velocities...). They can be seen as the equivalent of the characters (or bytes) for text, and the base vocabulary to the initial alphabet.

To train a tokenizer, MidiTok is backed by the Hugging Face `🤗tokenizers <https://github.com/huggingface/tokenizers>`_ Rust library allowing super-fast training and encoding. Thus, internally MidiTok represents basic tokens (from the base vocab) as characters (bytes). Essentially, a token will have three unique forms:

* The text describing the token itself, e.g. ``Pitch_58``, ``Position_4`` ...;
* An id as an integer, that will be fed to the model, e.g. ``65``. It corresponds to the index of the token in the vocabulary;
* A byte form, as a character or succession of characters, e.g. ``a`` or any `unicode character <https://en.wikipedia.org/wiki/List_of_Unicode_characters>`_ starting from the 33rd one (0x21).

A learned token will be represented by the succession of the unique characters of the base tokens it represent. You can access to several vocabularies to get the equivalents forms of tokens:

* ``vocab``: the base vocabulary, mapping token descriptions to their ids;
* ``vocab_model``: the vocabulary with learned tokens, mapping byte forms to their integer id;
* ``_vocab_base_byte_to_token``: mapping the base token byte forms to their string forms;
* ``_vocab_base_id_to_byte``: mapping the base token ids (integers) to their byte forms;
* ``_vocab_bpe_bytes_to_tokens``: mapping the byte forms of the complete vocab to their string forms, as a list of string;


Tokenizer models
------------------------

Byte Pair Encoding (BPE)
~~~~~~~~~~~~~~~~~~~~~~~~

`BPE <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_ is a compression algorithm that replaces the most recurrent token successions of a corpus, by newly created ones. It starts from a vocabulary containing tokens representing the initial alphabet of the modality of the data at hand, and iteratively counts the occurrences of each token successions, or bigrams, in the data, and merges the most recurrent one with a new token representing both of them, until the vocabulary reaches the desired size.

For instance, in the character sequence ``aabaabaacaa``, the sub-sequence ``aa`` occurs three times and is the most recurrent one. Learning BPE on this sequence would replace ``aa`` with a new symbol, e.g., ``d``, resulting in a compressed sequence ``dbdbdcd``. The latter can be reduced again by replacing the ``db`` subsequence, giving ``eedcd``. The vocabulary, which initially contained three characters (``a``, ``b`` and ``c``) now also contains ``d`` and ``e``. In practice BPE is learned on a corpus until the vocabulary reaches a target size.

Today in the NLP field, BPE is used with many tokenizers to build their vocabulary, as `it allows to encode rare words and segmenting unknown or composed words as sequences of sub-word units <https://aclanthology.org/P16-1162/>`_. The base initial vocabulary is the set of all the unique characters present in the data, which compose the words that are automatically learned as tokens by the BPE algorithm.

Unigram
~~~~~~~~~~~~~~~~~~~~~~~~

The `Unigram <https://aclanthology.org/P18-1007/>`_ algorithm serves the same purpose than BPE, but works in the other direction: it starts from a large vocabulary of byte successions (e.g. words) and substitute some of them with smaller pieces until the vocabulary reaches the desired size.

At each training step, Unigram compute the subword occurrence probabilities with the Expectation maximization (EM) algorithm and computes a loss over the training data and current vocabulary. For each token in the vocabulary, Unigram computes how much removing it would increase the loss. The tokens that increase the loss the least have the lowest impact on the overall data representation, and can be considered less important, and Unigram will remove them until the vocabulary reaches the desired size.

Note that the loss is computed over the whole training data and current vocabulary. This step is computationally expensive. Hence removing a single token per training step would require a significant amount of time. In practice, `n` percents of the vocabulary is removed at each step, with `n` being a hyperparameter to set.

Note that Unigram is not a deterministic algorithm: training a tokenizer twice with the same data and training parameter will likely result in similar vocabularies, but a few differences.
You can read more details on the loss computation in the `documentation of the tokenizers library <https://huggingface.co/learn/nlp-course/en/chapter6/7>`_.

The Unigram model supports the additional training arguments that can be provided as keyword arguments to the :py:func:`miditok.MusicTokenizer.train` method:

* ``shrinking_factor``: shrinking factor to used to reduce the vocabulary at each training step (default: 0.75);
* ``max_piece_length``: maximum length a token can reach (default in MidiTok: 50 if splitting ids per beats, 200 otherwise i.e. splitting ids per bars or no split);
* ``n_sub_iterations``: number of Expectation-Maximization algorithm iterations performed before pruning the vocabulary (default: 2).

Unigram is also implemented in the `SentencePiece <https://aclanthology.org/D18-2012/>`_ library.


WordPiece
~~~~~~~~~~~~~~~~~~~~~~~~

`WordPiece <https://ieeexplore.ieee.org/document/6289079>`_ is a subword-based algorithm very similar to BPE. The original implementation was never open-sourced by Google. The training procedure is known to be a variation of BPE. In `🤗tokenizers <https://github.com/huggingface/tokenizers>`_ (and so in MidiTok), BPE is used to create the vocabulary.

The difference with BPE lies in the way the bytes are tokenized after training: for a specific word to tokenize, WordPiece will look in the vocabulary if it is present. If so, there is nothing to do and the token id of the word can be used. Otherwise, it will decrement the word from its end until it finds a match in the vocabulary, and iteratively do the same for all the components ("pieces") of the word. The procedure is explained more in detail in the `Tensorflow documentation <https://www.tensorflow.org/text/guide/subwords_tokenizer#optional_the_algorithm>`_.

Intuitively, WordPiece tokenization is trying to satisfy two different objectives:

1. Tokenize the data into the least number of tokens as possible;
2. When a byte sequence needs to be split, it is split into tokens that have a maximum count in the training data.


..
    Commented, previous docs based on Hugging Face's course, which is actually incorrect: https://huggingface.co/learn/nlp-course/en/chapter6/6#tokenization-algorithm
    It counts the successions of tokens in the data, but instead of merging the pair with the highest count, WordPiece merges the one with the highest score computed as: :math:`\mathrm{score}_{ij} = \frac{ \mathrm{freq}_{ij} }{\mathrm{freq}_{i} \times \mathrm{freq}_{j}}` for two symbols :math:`i` and :math:`j`.

    Dividing the frequency of the succession of two symbols by the product of their frequency in the data allows to merge pairs where one token is less frequent. The most frequent pair will hence not necessarily be merged. Doing so, WordPiece tries to learn tokens while evaluating their impact.

    Another key difference with BPE is on the training procedure: WordPiece starts by computing all the pairs in the data, and then counts their frequencies. It will learn long words first, and splitting them in multiple subtokens when they do not exist in the vocabulary.

WordPiece features a ``max_input_chars_per_word`` attribute limiting the length of the "words", base tokens successions in MidiTok's case, it can process. Token successions with a length exceeding this parameter will be replaced by a ``unk_token`` token (MidiTok uses the padding token by default). You can set the ``max_input_chars_per_word`` in the keyword arguments of the :py:func:`miditok.MusicTokenizer.train` method, but the higher this parameter is, the slower the encoding-decoding will be. The number of base tokens for a music file is likely to go in the tens of thousands. As a result, **WordPiece should exclusively be used while splitting the token ids per bars or beats** in order to make sure that the lengths of the token successions remain below this limit.


Splitting the ids
------------------------

In MidiTok, we represent base tokens as bytes in order to use the Hugging Face `tokenizers <https://github.com/huggingface/tokenizers>`_ Rust library. The length of the token sequence of a music file can easily reach tens of thousands of tokens, depending on its number of tracks, notes in each track, and length in bars. As a result, if we convert this sequence in its byte form, we end with a one single very long word (one character per base token).
Using this single word to train the tokenizer is feasible (except for WordPiece), and doing so the tokenizer will learn new tokens representing successions of base tokens that can span across several bars and beats, and optimizes the sequence length reduction the most. However, learning tokens that can represent events starting and ending anywhere cannot ensure us to have tokens with musically relevant information. It could be seen as training a text tokenizer without splitting the text into words, thus learning tokens that also contain spaces between words or subwords.

MidiTok allows to split the token sequence into subsequences of bytes for each bar or beat, that will be treated separately by the tokenizer's model. This can be set by the ``encode_ids_split`` attribute of the tokenizer's configuration (:class:`miditok.classes.TokenizerConfig`). Doing so, the learned tokens will not span across different bars or beats. The splitting step is also performed before encoding token ids after that the training is done.
It is similar to the "pre-tokenization" step in the `Hugging Face tokenizers library <https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/components#pretokenizers>`_ which consists in splitting the input text into distinct words at spaces.

Training example
------------------------

..  code-block:: python

    from miditok import REMI, TokenizerConfig, TokSequence
    from copy import deepcopy

    tokenizer = REMI(TokenizerConfig(use_programs=True))
    paths_midis = list(Path("path", "to", "midis").glob('**/*.mid'))

    # Learns the vocabulary with BPE
    # Ids are split per bars by default
    tokenizer.train(
        vocab_size=30000,
        model="BPE",
        files_paths=paths_midis,
    )

    # Tokenize a MIDI file
    tokens = tokenizer(paths_midis[0])
    # Decode BPE
    tokens_no_bpe = tokenizer.decode_bpe(deepcopy(tok_seq))

Methods
------------------------

A tokenizer can be trained with the :py:func:`miditok.MusicTokenizer.train` method. After being trained, the tokenizer will automatically encode the token ids with its model when tokenizing music files.

Trained tokenizers can be saved and loaded back (:ref:`Save / Load a tokenizer`).

.. autofunction:: miditok.MusicTokenizer.train
    :noindex:

.. autofunction:: miditok.MusicTokenizer.encode_token_ids
    :noindex:

.. autofunction:: miditok.MusicTokenizer.decode_token_ids
    :noindex:
