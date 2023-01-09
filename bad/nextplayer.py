# pylint: disable=missing-module-docstring, wrong-import-position, import-error too-few-public-methods too-many-arguments

class NextPlayer:
    '''next player (may should belong to hanabi environment itself)'''
    def next_player(self, observation: dict) -> int:
        '''next player'''
        curr_player:int = observation['current_player']
        return (curr_player +1) % 2
