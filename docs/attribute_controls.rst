========================
Attribute Controls
========================

Attribute Controls are special tokens that allow to train a model in order to control music generation during inference. They work either at the track-level or bar-level and specifies specific attributes they featured. By being placed at the beginning of each bar or track in the token sequence, a *causal* model will condition the prediction of the next tokens based on them. At inference, these attribute control tokens can strategically be placed at the beginning of nex tracks or bars in order to condition the generated results.

Attribute controls are not compatible with "multi-vocabulary" (e.g. Octuple) or multitrack "one token stream" tokenizers.

To train tokenizers and models with attribute control tokens, you can use the :class:`miditok.TokTrainingIterator` and :class:`miditok.pytorch_data.DatasetMIDI` respectively.

.. automodule:: miditok.attribute_controls
    :members:

Using custom attribute controls
-------------------------------

You can easily add your own attribute controls to an existing tokenizer using the :py:func:`miditok.MusicTokenizer.add_attribute_control` method. You attribute control must subclass either the :class:`miditok.attribute_controls.AttributeControl` (track-level) or the :class:`miditok.attribute_controls.BarAttributeControl` classes and implement the attribute computation method.
