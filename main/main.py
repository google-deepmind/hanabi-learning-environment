# pylint: disable=missing-module-docstring, wrong-import-position, no-name-in-module, unused-variable, unused-variable

import os
import random
import sys
import numpy as np
import tensorflow as tf


currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.self_play import SelfPlay
from bad.train_batch import TrainBatch
from bad.action_network import ActionNetwork

def main() -> None:
    '''main'''
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size: int = 50
    episodes_running = 100
    discount = 0.95

    print(f'welcome to bad agent with tf version: {tf.__version__}')
    print(f'running {episodes_running} episodes')

    network: ActionNetwork = ActionNetwork()

    train_batch = TrainBatch(network)
    training_result = train_batch.run(batch_size=batch_size, discount=discount)

    self_play = SelfPlay(network)
    self_play.run(episodes_running)

    print("finish with everything")
if __name__ == "__main__":
    main()
