========================
Data augmentation
========================

Data augmentation is a technique to artificially increases the size of a dataset by applying various transformations on to the existing data. These transformations consist in altering one or several attributes of the original data. In the context of images, they can include operations such as rotation, scaling, cropping or color adjustments. This is more tricky in the case of natural language, where the meaning of the sentences can easily diverge following how the text is modified, but some techniques such as paraphrase generation or back translation can fill this purpose.

The purpose of data augmentation is to introduce variability and diversity into the training data without collecting additional real-world data. Data augmentation can be important and increase a model's learning and generalization, as it exposes it to a wider range of variations and patterns present in the data. In turn it can increases its robustness and decrease overfitting.

MidiTok allows to perform data augmentation, on the MIDI level and token level. Transformations can be made by increasing the values of the velocities and durations of notes, or by shifting their pitches by octaves. Data augmentation is highly recommended to train a model, in order to help a model to learn the global and local harmony of music. In large datasets such as the `Lakh <https://colinraffel.com/projects/lmd/>`_ or `Meta MIDI <https://zenodo.org/records/5142664>`_ datasets, MIDI files can have various ranges of velocity, duration values, and pitch. By augmenting the data, thus creating more diversified data samples, a model can better generalize learning the melody, harmony and music features rather than learning specific recurrent token successions.

.. automodule:: miditok.data_augmentation
    :members:
