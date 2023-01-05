# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.trainepoch import TrainEpoch
from bad.actionnetwork import ActionNetwork

class Runner:
    '''runner'''
    def train(self, episodes: int) -> None:
        '''train'''
        network: ActionNetwork = ActionNetwork()
        network.build()

        for episode in range(episodes):
            print(f"begin training for {episode}")
            train: TrainEpoch = TrainEpoch(network)
            train.run(network)

    def run(self, episodes: int) -> None:
        '''run'''
        for episode in range(episodes):
            # observations = self.environment.reset()
            print(f"begin running episode: {episode}")
