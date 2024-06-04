========================
Hugging Face Hub
========================

What is the Hugging Face hub
---------------------------------

The `Hugging Face Hub <https://huggingface.co>`_ is a model and dataset sharing platform which is widely used in the AI community. It allows to freely upload, share and download models and datasets, directly in your code in a very convenient way. Its interactions rely on an open-source Python package named `huggingface_hub <https://github.com/huggingface/huggingface_hub>`_. As it works seamlessly in the Hugging Face ecosystem, especially the `Transformers <https://huggingface.co/docs/transformers/index>`_ or `Diffusers <https://huggingface.co/docs/diffusers/index>`_ libraries, it stood out and became one of the preferred way to openly share and download models.

Now when downloading a Transformer model, you will need to also download its associated tokenizer to be able to "dialog" with it. Likewise, if you want to share one of your models, you will need to share its tokenizer too for people to be able to use it. MidiTok allows you to push and download tokenizers in similar way to what is done in the Hugging Face Transformers library.

How MidiTok interoperates with the hub
------------------------------------------

Internally, MidiTok relies on the ``huggingface_hub.ModelHubMixin`` component. It implements the same methods commonly used in the Hugging Face ecosystem. Note that:

* :py:func:`miditok.MusicTokenizer.save_pretrained` is equivalent to calling :py:func:`miditok.MusicTokenizer.save_params`;
* :py:func:`miditok.MusicTokenizer.from_pretrained` can be used to load tokenizers whether from the Hugging Face hub or from a file on your local filesystem;
* for :py:func:`miditok.MusicTokenizer.save_pretrained` and :py:func:`miditok.MusicTokenizer.push_to_hub`, you can ignore the ``config`` argument which is meant to be used with models (not applicable for tokenizers);
* you can give a ``filename`` keyword argument with the :py:func:`miditok.MusicTokenizer.save_pretrained` and :py:func:`miditok.MusicTokenizer.from_pretrained` methods to use a specific tokenizer configuration file name, otherwise the default one will be used (``tokenizer.json``).

.. autofunction:: miditok.MusicTokenizer.from_pretrained
    :noindex:

.. autofunction:: miditok.MusicTokenizer.save_pretrained
    :noindex:

.. autofunction:: miditok.MusicTokenizer.push_to_hub
    :noindex:

Example
------------------------

..  code-block:: python

    from miditok import REMI, TokSequence
    from copy import deepcopy

    tokenizer = REMI()  # using defaults parameters (constants.py)
    hf_token = "your_hf_token"  # to create on huggingface.co

    # Train the tokenizer with BPE
    tokenizer.train(
        vocab_size=30000,
        files_paths=list(Path("path", "to", "midis").glob("**/*.mid")),
    )

    # Push the tokenizer to the HF hub
    tokenizer.push_to_hub("YourUserName/model-name", private=True, token=hf_token)

    # Recreates it from the configuration saved on the hub
    tokenizer2 = REMI.from_pretrained("YourUserName/model-name", token=hf_token)
    assert tokenizer == tokenizer2
