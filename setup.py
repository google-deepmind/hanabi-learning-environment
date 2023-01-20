# pylint: disable=missing-module-docstring
from setuptools import find_packages
from skbuild import setup

setup(
    name='ws2223-group7-hanabi-learning-environment_bad',
    version='0.0.5',
    description='ws2223-group7 Learning environment for the game of hanabi bad agent.',
    long_description='ws2223-group7 Learning environment for the game of hanabi bad agent.',
    long_description_content_type="text/markdown",
    author='ws2223-group7/hanabi-learning-environment_bad',
    packages=find_packages(),
    install_requires=['cffi'],
    python_requires=">=3.6",
    classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
    ],
    entry_points={
    'console_scripts': [
        'ws2223-group7-hanabi-learning-environment-bad = bad.main:main',
    ],
    }
)
