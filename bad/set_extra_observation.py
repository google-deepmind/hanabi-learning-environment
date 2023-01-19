# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import numpy as np

class SetExtraObservation:
    '''set extra observation'''
    def set_extra_observation(self, hanabi_observation:dict, \
        last_action: int, max_action:int, legal_moves: list) -> None:
        '''set extra observation'''
        hanabi_observation['last_action'] = last_action
        hanabi_observation['max_action'] = max_action

        legal_moves_int = np.empty(0, int)
        for legal_move in legal_moves:
            legal_moves_int = np.append(legal_moves_int, legal_move)

        hanabi_observation['legal_actions'] = legal_moves_int
