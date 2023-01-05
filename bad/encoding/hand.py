# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.nextplayer import NextPlayer

class Hand():
    '''hand'''
    def __init__(self, observation: dict) -> None:
        curr_player_id:int = observation['current_player']
        
        next_player = NextPlayer()
        next_player_id = next_player.next_player(observation)
