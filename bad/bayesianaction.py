# pylint: disable=missing-module-docstring too-few-public-methods)
import numpy as np


class BayesianAction:
    '''Bayesian Action'''
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions

    def decode_action(self, legal_moves:np.ndarray) -> int:
        '''returns a choice'''
        legal_moves_int = legal_moves.tolist()
        all_actions = self.actions.copy()

        for action in range(0, all_actions.size):
            exists: bool = legal_moves_int.count(action) > 0
            if not exists:
                all_actions[action] = -float('inf')

        result = np.argmax(all_actions, axis=0)
        return int(result)
