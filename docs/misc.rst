========================
Misc
========================

Retro-compatibility with Python 3.7
-------------------------------------

As of the version 2.1.8 of MidiTok, we drop support for the versions of Python previous to 3.8. Even though Python 3.7 reached its end of life in June 2023, we were committed to still support it till we could cover the tests with no additional cost and that no feature from newer versions were required. In MidiTok 2.1.8, we switched from the legacy ``setup.py`` project metadata to the ``pyproject.toml`` format, allowing to specify package versions in one place. With this, we required to use the ``importlib`` library to dynamically parse package versions within MidiTok. ``importlib`` does not come with Python 3.7 and has a slightly different usage. In order to reduce our software debt, it was decided to drop support or Python 3.7.

You can still use the latest versions of MidiTok with Python 3.7, but it will require you to first manually install ``importlib``:

..  code-block:: bash
    pip install importlib-metadata

Then to edit its import in the `constants.py <https://github.com/Natooz/MidiTok/blob/main/miditok/constants.py>`_ file:

..  code-block:: python
    # Replace:
    # from importlib.metadata import version
    # with:
    from importlib_metadata import version

And finally to install MidiTok locally:

..  code-block:: bash
    pip install .
