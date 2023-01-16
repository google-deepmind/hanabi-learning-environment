# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module
import random
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.trainepoch import TrainEpoch

class Runner:
    '''runner'''
    def __init__(self) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, episodes: int, batch_size:int) -> None:
        '''train'''
        for episode in range(episodes):
            print(f"begin training for {episode}")
            train_epoch = TrainEpoch()
            train_epoch.train(batch_size)

    def run(self, episodes: int) -> None:
        '''run'''
        for episode in range(episodes):
            # observations = self.environment.reset()
            print(f"begin running episode: {episode}")
