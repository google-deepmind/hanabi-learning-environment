# pylint: disable=missing-module-docstring too-few-public-methods, pointless-string-statement
import numpy as np
import tensorflow_probability as tfp


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

        cat = tfp.distributions.Categorical(logits=all_actions)
        next_action = int(cat.sample().numpy().T)

        return int(next_action)
