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
    batch_size: int = 1
    episodes_training = 1
    episodes_running = 1000

    print(f'welcome to bad agent with tf version: {tf.__version__}')
    print(f'running {episodes_running} episodes')

    runner = Runner()
    runner.train(episodes=episodes_training, batch_size=batch_size)
    runner.self_play(episodes_running)

    print("finish with everything")
if __name__ == "__main__":
    main()
