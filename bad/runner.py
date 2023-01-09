# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.trainepoch import TrainEpoch

class Runner:
    '''runner'''

    def train(self, episodes: int) -> None:
        '''train'''
        for episode in range(episodes):
            print(f"begin training for {episode}")
            train_epoch = TrainEpoch()
            train_epoch.train()

    def run(self, episodes: int) -> None:
        '''run'''
        for episode in range(episodes):
            # observations = self.environment.reset()
            print(f"begin running episode: {episode}")
