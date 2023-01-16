# pylint: disable=missing-module-docstring too-few-public-methods)
import random
import numpy as np


class BayesianAction:
    '''Bayesian Action'''
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions

    def random_action(self) -> int:
        '''returns a choice'''
        return random.choice(range(0, self.actions.size))
