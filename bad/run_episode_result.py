# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument, unnecessary-pass, consider-using-enumerate
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.rl_env import HanabiEnv

class RunEpisodeResult:
    '''run episode result'''
    def __init__(self,number:int, reward:int, hanabi_environment: HanabiEnv) -> None:
        self.number = number
        self.reward  = reward
        self.hanabi_environment = hanabi_environment
