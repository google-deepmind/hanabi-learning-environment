import tensorflow as tf

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.runner import Runner

def main() -> None:
    print(f'welcome to bad agent with tf version: {tf.__version__}')
    runner = Runner()
    runner.train(1)
    runner.run(10)

if __name__ == "__main__":
  main()
