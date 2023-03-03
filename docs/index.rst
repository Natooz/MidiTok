.. MidiTok documentation master file, created by
   sphinx-quickstart on Sat Feb  4 20:52:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MidiTok's documentation!
=========================================

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/logo.png?raw=true
  :width: 600
  :alt:

**MidiTok** is a Python package for MIDI file tokenization, presented at the ISMIR 2021 LBDs `(paper) <https://archives.ismir.net/ismir2021/latebreaking/000005.pdf>`_.
It converts MIDI files to sequences of tokens ready to be fed to sequential Deep Learning models such as Transformers.

MidiTok features most known MIDI :ref:`tokenizations`, and is built around the idea that they all share common methods. It properly pre-process MIDI files, and supports :ref:`Byte Pair Encoding (BPE)`.
`Github repository <https://github.com/Natooz/MidiTok>`_

Installation
==================

..  code-block:: bash

    pip install miditok

MidiTok uses `MIDIToolkit <https://github.com/YatingMusic/miditoolkit>`_ and `Mido <https://github.com/mido/mido>`_ to read and write MIDI files, and BPE is backed by `Hugging Face 🤗tokenizers <https://github.com/huggingface/tokenizers>`_ for super fast encoding.

Citation
==================

If you use MidiTok for your research, a citation in your manuscript would be gladly appreciated. ❤️

You can also find BibTeX :ref:`citations` of tokenizations.

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
   midi_tokenizer
   examples
   tokenizations
   bpe
   data_augmentation
   utils
   citations

