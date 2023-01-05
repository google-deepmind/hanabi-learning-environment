
# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import unittest
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.nextplayer import NextPlayer


class TestNextPlayer(unittest.TestCase):
    '''test next player'''
    def test_next_player_is_one(self):
        ''' test next player '''
        observation = {}
        observation['current_player'] = 0

        next_player = NextPlayer()
        next_player_id = next_player.next_player(observation)

        self.assertEqual(next_player_id, 1)

    def test_next_player_is_zero(self):
        ''' test next player '''
        observation = {}
        observation['current_player'] = 1

        next_player = NextPlayer()
        next_player_id = next_player.next_player(observation)

        self.assertEqual(next_player_id, 0)
