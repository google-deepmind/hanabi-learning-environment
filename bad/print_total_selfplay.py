# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods, consider-using-enumerate

class PrintTotalSelfPlay:
    '''print total selfplay'''
    def __init__(self, episodes:int, total_reward:int, max_reward: int, perfect_games: int) -> None:
        '''init'''
        self.total_reward = total_reward
        self.max_reward = max_reward
        self.episodes = episodes
        self.perfect_games = perfect_games

    def print(self) -> None:
        '''print'''
        print(f"Total Reward: {self.total_reward}")
        print(f"Max  Reward: {self.max_reward}")
        print(f"Avg. Reward: {format(self.total_reward/(self.episodes+1),'.3f')}")
        print(f"Perfecet Games: {format(self.perfect_games)}")
