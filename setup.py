from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="miditok",
    author="Nathan Fradet",
    url="https://github.com/Natooz/MidiTok",
    packages=find_packages(exclude=("tests",)),
    version="2.1.8",
    license="MIT",
    description="MIDI / symbolic music tokenizers for Deep Learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "midi",
        "tokenization",
        "music",
        "mir",
    ],
    install_requires=[
        "numpy>=1.19",
        "miditoolkit",  # TODO >=v1.0.1
        "tqdm",
        "tokenizers>=0.13.0",
        "huggingface_hub>=0.16.4",
        "scipy",  # needed for miditoolkit TODO remove when miditoolkit v1.0.1
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
