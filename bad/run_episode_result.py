# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument, unnecessary-pass, consider-using-enumerate
import sys
import os

import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.rl_env import HanabiEnv

class RunEpisodeResult:
    '''run episode result'''
    def __init__(self,number:int, reward:int, hanabi_environment: HanabiEnv, \
        agent_step_times: np.ndarray) -> None:
        self.number = number
        self.reward  = reward
        self.hanabi_environment = hanabi_environment
        self.agent_step_times = agent_step_times
