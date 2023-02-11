# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.bayesian_action_result import BayesianActionResult
from bad.encoding.observation import Observation


class Buffer:
    '''buffer'''
    def __init__(self) -> None:
        self.hanabi_observation: list[dict] = []
        self.observation: list[Observation] = []
        self.actions: list[BayesianActionResult] = []
        self.rewards: list[int] = []

    def append(self, hanabi_observation: dict, observation: Observation, \
    action_result: BayesianActionResult, reward: int) -> None:
        '''add'''
        self.hanabi_observation.append(hanabi_observation)
        self.observation.append(observation)
        self.actions.append(action_result)
        self.rewards.append(reward)
