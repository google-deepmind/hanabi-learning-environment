# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import os
import sys
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.runner import Runner

def main() -> None:
    '''main'''
    print(f'welcome to bad agent with tf version: {tf.__version__}')
    runner = Runner()
    runner.train(1)
    runner.run(10)

if __name__ == "__main__":
    main()
