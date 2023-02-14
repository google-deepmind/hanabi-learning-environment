# pylint: disable=missing-module-docstring, wrong-import-position, no-name-in-module, too-few-public-methods

class Constants:
    '''constants'''

    def __init__(self) -> None:
        '''init'''
        self.num_ply = None
        self.num_cards = None
        self.environment_name = 'Hanabi-Group-7'

    def update(self, hanabi_env) -> None:
        '''update'''
        # rank infos needs to update munally 
        self.max_rank = 4 
        self.num_cards_per_rank = [3, 2, 2, 1]

        self.num_ply = len(hanabi_env['player_observations'])
        self.num_hand_cards = len(hanabi_env['player_observations'][0]
                                       ['observed_hands'][0])
        
        self.colors = hanabi_env['player_observations'][1]['fireworks'].keys()
        