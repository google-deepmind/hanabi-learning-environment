from skbuild import setup

setup(
    name='ws2223-group7-hanabi-learning-environment',
    version='0.0.2',    
    description='ws2223-group7 Learning environment for the game of hanabi.',
    long_description='ws2223-group7 Learning environment for the game of hanabi.',
    long_description_content_type="text/markdown",
    author='ws2223-group7/hanabi-learning-environment',
    packages=['hanabi_learning_environment', 'hanabi_learning_environment.agents'],
    install_requires=['cffi'],
    python_requires=">=3.6",
    classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
    ]    
)
