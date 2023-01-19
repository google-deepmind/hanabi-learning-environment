# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument, unnecessary-pass, consider-using-enumerate
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.bad_agent import BadAgent
from bad.actionnetwork import ActionNetwork
from bad.policy import Policy
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation

class RunEpisode:
    '''runs an episode'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.policy = Policy(network)
        players:int = 2
        self.hanabi_environment = rl_env.make('Hanabi-Full', players, \
            pyhanabi.AgentObservationType.CARD_KNOWLEDGE.CARD_KNOWLEDGE)
        self.agents = [BadAgent(self.policy, self.hanabi_environment), \
            BadAgent(self.policy, self.hanabi_environment)]

    def run(self, episode_number: int) -> None:
        '''run'''
        print(f"run episode {episode_number}")

        hanabi_observation = self.hanabi_environment.reset()
        # one more move because of no-action move on the beginning
        #fake an action that does not exists
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based

        seo = SetExtraObservation()

        done = False
        while not done:
            for agent_id in range(len(self.agents)):

                seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
                    self.hanabi_environment.state.legal_moves_int())

                result = self.agents[agent_id].act(hanabi_observation)
                hanabi_observation = result.observation_after_step
                done = result.done
                if done:
                    break
