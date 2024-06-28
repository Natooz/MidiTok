.. MidiTok documentation master file, created by
   sphinx-quickstart on Sat Feb  4 20:52:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MidiTok's documentation!
=========================================

.. image:: /assets/miditok_logo_stroke.png
  :width: 600
  :alt:

**MidiTok** is a Python package for MIDI file tokenization, introduced at the ISMIR 2021 LBDs `(paper) <https://archives.ismir.net/ismir2021/latebreaking/000005.pdf>`_.
It tokenize symbolic music files (MIDI, abc), i.e. convert them into sequences of tokens ready to be fed to models such as Transformer, for any generation, transcription or MIR task.
MidiTok features most known MIDI :ref:`tokenizations`, and is built around the idea that they all share common methods. Tokenizers can be trained with BPE, Unigram or WordPiece (:ref:`Training a tokenizer`) and be push to and pulled from the Hugging Face hub!

Installation
==================

..  code-block:: bash

    pip install miditok

MidiTok uses `symusic <https://github.com/Yikai-Liao/symusic>`_ to read and write MIDI files, and tokenizer training is backed by the `Hugging Face 🤗tokenizers <https://github.com/huggingface/tokenizers>`_ for super fast encoding.

Citation
==================

If you use MidiTok for your research, a citation in your manuscript would be gladly appreciated. ❤️

You can also find in this documentation BibTeX :ref:`citations` of related research works.

..  code-block:: bib

    @inproceedings{miditok2021,
        title={{MidiTok}: A Python package for {MIDI} file tokenization},
        author={Fradet, Nathan and Briot, Jean-Pierre and Chhel, Fabien and El Fallah Seghrouchni, Amal and Gutowski, Nicolas},
        booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference},
        year={2021},
        url={https://archives.ismir.net/ismir2021/latebreaking/000005.pdf},
    }

Contents
==================

.. toctree::
   :maxdepth: 2
   :caption: Bases of Music and AI

   music_formats
   midi
   sequential_models

.. toctree::
   :maxdepth: 2
   :caption: MidiTok

   tokenizing_music_with_miditok
   configuration
   tokenizations
   attribute_controls
   train
   hf_hub
   pytorch_data
   data_augmentation
   utils

.. toctree::
   :maxdepth: 2
   :caption: Others

   examples
   citations

.. toctree::
   :hidden:
   :caption: Project Links

   GitHub <https://github.com/Natooz/MidiTok>
   PyPi <https://pypi.org/project/miditok/>
