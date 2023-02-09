.. MidiTok documentation master file, created by
   sphinx-quickstart on Sat Feb  4 20:52:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MidiTok's documentation! (WIP)
=========================================

.. image:: https://github.com/Natooz/MidiTok/blob/assets/assets/logo.png?raw=true
  :width: 600
  :alt:

**MidiTok** is a Python package for MIDI file tokenization, presented at the `ISMIR 2021 LBDs <https://archives.ismir.net/ismir2021/latebreaking/000005.pdf>`_.
Tt converts MIDI files to sequences of tokens (integers) ready to be fed to sequential Deep Learning models such as Transformers.

MidiTok features most known MIDI tokenization strategies, and is built around the idea that they all share common methods. It properly pre-process MIDI files, and supports Byte Pair Encoding (BPE).

Installation
==================

..  code-block:: bash
    :caption: You can install it with PIP

    pip install miditok

Contents
==================

.. toctree::
   midi_tokenizer
   examples
   tokenizations
   bpe

