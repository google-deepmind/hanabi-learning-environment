#!/bin/bash

pip install wheel setuptools scikit-build cffi ninja cmake
pip install wheel cmake-build-extension

pip wheel . --no-deps -w wheelhouse/
unzip ./wheelhouse/*.whl  -d ./wheelhouse

yes | cp -rf ./wheelhouse/hanabi_learning_environment/libpyhanabi.so ./hanabi_learning_environment

rm -r wheelhouse