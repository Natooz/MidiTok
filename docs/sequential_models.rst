.. _sequential-models-label:

===================================
Sequential models and tokens
===================================

This page introduces the basic concepts of sequential models, which are often called "language models" as commonly use for natural language, which can be used with MidiTok to be trained on music data.


Sequential models
----------------------------

We qualify as sequential model and model that takes as input **sequences of discrete elements**. `RNN <http://www.cs.toronto.edu/~hinton/absps/pdp8.pdf>`_\, `Long Short Term Memory (LSTM) <https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext>`_ and `Transformers <https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`_ fall into this category. As a general rule, the operation of these models noted :math:`p_\theta` can be formulated as :math:`p_\theta (\mathbf{x}) = y` where :math:`\mathbf{x} \in \mathbb{N}^n` is a sequence of :math:`\mathbb{N}^n` elements (integers here) and :math:`y` can either be a scalar or a sequence. The common feature of these is that :math:`y` **is conditioned on all the elements from** :math:`\mathbf{x}`.

.. _transformer-label:

..  figure:: /assets/transformer.png
    :alt: Schema of a Transformer model
    :class: with-shadow
    :width: 500px

    Schema of a "seq2seq" Transformer model.

A sequential model can be "seq2seq", "encoder-only" or "decoder-only".
seq2seq means that the model is composed of an encoder and decoder. The model's encoder processes an input sequence into intermediate **hidden states**, which condition the decoder that **autoregressively** generate the output sequence. This architecture is commonly used for translation tasks where the input sequence is in one language and the decoder generates its translation in another one.

In a seq2seq configuration, the encoder is usually **bi-directional**, meaning that the all the output hidden states are conditioned on all the input elements, whereas the decoder is **causal**, meaning that the logits of a position :math:`t` are conditioned only on the input elements at positions :math:`\leq t`, i.e. the previous ones.

An encoder-only model (e.g. `BERT <https://arxiv.org/abs/1810.04805>`_\) is more useful for non-generative tasks, e.g. classification. On the other hand, a decoder-only model is usually designed to generate content. As each position is conditioned on the previous ones, the model is usually trained with **teacher forcing** to predict the next element. Consequently, it can be used to generate content **autoregressively**, i.e. one element after another on :math:`n` iterations by reinjecting the element generated at a given iteration to the end of the input sequence of the next one.


Tokens and vocabulary
----------------------------

This section focuses more specifically on the nature of the inputs of the models.

Until now, we referred to the sequences as holding "elements" representing discrete attributes of the data. These elements are commonly called **tokens**, and **are fed to a model as integers**. For natural language, these tokens can represent words or parts of words. Consequently, sentence can then be tokenized into a sequence of tokens representing the words and punctuation. For symbolic music, tokens can represent the values of the note attributes (pitch, velocity, duration) or time events. The conversion of raw data to tokens is done by a **tokenizer**, which reads it and serializes it into sequences of tokens from its vocabulary.

The **vocabulary** of a tokenizer is the finite set of all distinct known tokens. For natural language, it represent the set of words, subwords, punctuations and unicode characters. **Each token is associated to a unique id**, its index in the vocabulary, which is fed to a model. A vocabulary is usually (as in MidiTok) a dictionary acting a lookup table linking tokens (their text forms) to their ids (integer form).


Embeddings
----------------------------

This section introduces the notion of embedding, sometimes called *embedding vector* or *word embedding*.

Vocabularies are often made of thousands of tokens, each of them having a whole variety of meanings and significations. In order for a sequential model to efficiently process them, it must be able to capture their semantic information and features. This step is handled thanks to **embeddings**.

An embedding :math:`\mathbf{e}^d` is a vector of :math:`d` dimensions, which represent the semantic information of the associated token. The embeddings are **contextually learned** by the model during training, meaning their position are adjusted conditionally to the context in which they are found in the data. Embeddings with similar semantics/meanings will be closer in the **continuous embedding space** of the model than embeddings with no related meanings. They offer a way to the model to capture the semantic of words across these dimensions.

..  figure:: /assets/embeddings.png
    :alt: Embedding space.
    :class: with-shadow
    :width: 500px

    Visualization of an embedding space reduced in 2 dimensions with `TSNE <https://www.jmlr.org/papers/v9/vandermaaten08a.html>`_\.

The embeddings are actually the real input of a sequential model. Each token acts as an index for the model's embedding matrix. In :ref:`transformer-label`, the first operation consist in indexing this matrix with the token ids to get their embeddings which are then processed by the model.

MidiTok allows you to leverage the features of model embeddings by training the tokenizer (:ref:`training-tokenizer-label`).
