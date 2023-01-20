# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, unnecessary-pass, too-few-public-methods

class BadAgentActingResult:
    ''''bad agent acting result'''
    def __init__(self, observation_after_step: dict, done: bool, reward:int) -> None:
        self.observation_after_step = observation_after_step
        self.done = done
        self.reward = reward
