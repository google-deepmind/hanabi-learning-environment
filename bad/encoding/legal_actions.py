# pylint: disable=missing-module-docstring, wrong-import-position, import-error, too-few-public-methods too-many-arguments
import numpy as np
import tensorflow as tf

class LegalActions:
    '''legal moves'''
    def __init__(self, observation: dict) -> None:
        legal_actions:list = observation['legal_actions']
        max_action:int = observation ['max_action']
        self.vector = np.empty(0, int)
        for one_action in range(max_action):
            legal_actions_list = legal_actions.tolist()
            exists: bool = legal_actions_list.count(one_action) > 0
            self.vector = np.append(self.vector, self.get_encoded_action_exists(exists))

    def get_encoded_action_exists(self, exists: bool) -> np.ndarray:
        '''get encoded action exists'''
        if exists:
            return tf.keras.utils.to_categorical(1, num_classes=2, dtype=int)

        return tf.keras.utils.to_categorical(0, num_classes=2, dtype=int)
