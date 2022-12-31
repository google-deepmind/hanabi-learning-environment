# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.bad_agent import BadAgent
from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi

class Runner:
    '''runner'''
    def __init__(self):
        players = 2
        self.environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.SEER)
        self.environment.reset()
        self.agent_class = BadAgent

    def train(self, episodes: int) -> None:
        '''train'''
        for episode in range(episodes):
            print(f"begin training for {episode}")

    def run(self, episodes: int) -> None:
        '''run'''
        for episode in range(episodes):
            # observations = self.environment.reset()
            print(f"begin running episode: {episode}")
