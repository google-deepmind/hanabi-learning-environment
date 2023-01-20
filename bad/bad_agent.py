# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment.rl_env import Agent, HanabiEnv
from bad.bad_agent_acting_result import BadAgentActingResult
from bad.policy import Policy
from bad.encoding.observationconverter import ObservationConverter

class BadAgent(Agent):
    ''' bad agent '''
    def __init__(self, policy: Policy, hanabi_environment: HanabiEnv) -> None:
        self.policy = policy
        self.hanabi_environment = hanabi_environment
        self.observation_converter: ObservationConverter = ObservationConverter()

    def act(self, observation) -> BadAgentActingResult:
        '''act'''
        bad = self.policy.get_action(self.observation_converter.convert(observation))
        next_action = bad.decode_action(self.hanabi_environment.state.legal_moves_int())
        observation_after_step, reward, done, _ = self.hanabi_environment.step(next_action)
        return BadAgentActingResult(observation_after_step, done, int(reward))

    def reset(self, config):
        print('reset')
