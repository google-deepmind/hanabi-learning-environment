import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.rl_env import Agent
from bad.actionnetwork import ActionNetwork

class BadAgent(Agent):
    def __init__(self) -> None:
        self.actionNetwork = ActionNetwork()
        self.actionNetwork.build()