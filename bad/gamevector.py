import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)


class GameVector:

    def __init__(self, lifeTokensLeft : np.ndarray, hintTokensLeft: np.ndarray):
        self.lifeTokensLeft : np.ndarray = lifeTokensLeft
        self.hintTokensLeft : np.ndarray = hintTokensLeft
        """
        by drl_for_hanabi
        Turns a HLE observation into a vector.
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs).
        Assumes: there are 2 players.

        obs_vec info:
        [
        life tokens left, thermometer, for example: 110 means two tokens left (length 3)
        hint tokens left, for example: 11111100 means six left (length 8)

        firework rank for red, for example: 11000 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 5)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, one-hot, for example: 10000 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 10)

        other player color card index 0, one-hot, for example: 01000 is yellow
        other player rank  card index 0, one-hot, for example: 10000 means rank 1
        other player card index 1 (color & rank)
        other player card index 2 (color & rank)
        other player card index 3 (color & rank)
        other player card index 4 (color & rank)   (5 times length 10)

        discarded cards for red, for example 110 00 10 00 0 means two red 1's and one red 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 10)
        ]

        total vector length = 3 + 8 + 25 + 50 + 50 + 50 = 186
    """
