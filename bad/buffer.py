# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
class Buffer:
    '''buffer'''
    def __init__(self) -> None:
        self.hanabi_observation: list[dict] = []
        self.actions: list[int] = []
        self.rewards: list[int] = []

    def add(self, hanabi_observation: dict, next_action: int, reward: int) -> None:
        '''add'''
        self.hanabi_observation.append(hanabi_observation)
        self.actions.append(next_action)
        self.rewards.append(reward)
