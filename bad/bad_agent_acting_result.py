# pylint: disable=missing-module-docstring, wrong-import-position, no-member, no-name-in-module, too-few-public-methods

class BadAgentActingResult:
    ''''bad agent acting result'''
    def __init__(self, observation_after_step: dict, done: bool, reward:int) -> None:
        self.observation_after_step = observation_after_step
        self.done = done
        self.reward = reward
