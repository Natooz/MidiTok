========================
Attribute Controls
========================

Attribute Controls are special tokens that allow to train a model in order to control music generation during inference. They work either at the track-level or bar-level and specifies specific attributes they featured. By being placed at the beginning of each bar or track in the token sequence, a *causal* model will condition the prediction of the next tokens based on them. At inference, these attribute control tokens can strategically be placed at the beginning of nex tracks or bars in order to condition the generated results.

.. automodule:: miditok.attribute_controls
    :members:

Using custom attribute controls
-------------------------------

You can easily add your own attribute controls to an existing tokenizer using the :py:func:`miditok.MusicTokenizer.add_attribute_control` method. You attribute control must subclass either the :class:`miditok.attribute_controls.AttributeControl` (track-level) or the :class:`miditok.attribute_controls.BarAttributeControl` classes and implement the attribute computation method.
