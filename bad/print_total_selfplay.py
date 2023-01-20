# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument, unnecessary-pass, consider-using-enumerate

class PrintTotalSelfPlay:
    '''print total selfplay'''
    def __init__(self, episodes:int, total_reward:int, max_reward: int) -> None:
        '''init'''
        self.total_reward = total_reward
        self.max_reward = max_reward
        self.episodes = episodes

    def print(self) -> None:
        '''print'''
        print(f"Total Reward {self.total_reward}")
        print(f"Max  Reward: {self.max_reward}")
        print(f"Avg. Reward: {format(self.total_reward/(self.episodes+1),'.3f')}")
