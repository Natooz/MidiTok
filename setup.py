from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='miditok',
    author='Nathan Fradet',
    url='https://github.com/Natooz/MidiTok',
    packages=find_packages(exclude=("tests",)),
    version='1.4.1',
    license='MIT',
    description='A convenient MIDI tokenizer for Deep Learning networks, with multiple encoding strategies',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformer',
        'midi',
        'tokenization',
        'music',
        'mir'
    ],
    install_requires=[
        'numpy>=1.19,<1.24',
        'miditoolkit>=0.1.16',
        'tqdm'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ],
)
