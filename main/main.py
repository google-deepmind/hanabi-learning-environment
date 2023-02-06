# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-name-in-module

import os
import sys
import tensorflow as tf


currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.runner import Runner
from bad.train_batch import TrainBatch

def main() -> None:
    '''main'''
    batch_size: int = 500
    episodes_running = 100

    print(f'welcome to bad agent with tf version: {tf.__version__}')
    print(f'running {episodes_running} episodes')

    train_batch = TrainBatch()
    training_result = train_batch.run(batch_size=batch_size)

    runner = Runner()
    runner.self_play(episodes_running)

    print("finish with everything")
if __name__ == "__main__":
    main()
