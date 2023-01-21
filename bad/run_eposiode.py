# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-few-public-methods unused-variable, no-method-argument, unnecessary-pass, consider-using-enumerate, too-many-function-args
import sys
import os
import time

import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from hanabi_learning_environment import pyhanabi, rl_env
from bad.bad_agent import BadAgent
from bad.action_network import ActionNetwork
from bad.policy import Policy
from bad.set_extra_observation import SetExtraObservation
from bad.run_episode_result import RunEpisodeResult

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
        self.seo = SetExtraObservation()

    def run(self, episode_number: int) -> RunEpisodeResult:
        '''run'''
        print(f"run episode {episode_number}")

        hanabi_observation = self.hanabi_environment.reset()
        # one more move because of no-action move on the beginning
        #fake an action that does not exists
        max_moves: int = self.hanabi_environment.game.max_moves() + 1
        max_actions = max_moves + 1 # 0 index based

        episode_reward = 0
        done = False
        agent_step_times = np.empty(0, float)
        while not done:
            for agent_id in range(len(self.agents)):

                self.seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
                    self.hanabi_environment.state.legal_moves_int())
                start_time = time.time()
                result = self.agents[agent_id].act(hanabi_observation)
                end_time = time.time()
                agent_step_times = np.append(agent_step_times, end_time - start_time)
                hanabi_observation = result.observation_after_step
                done = result.done
                episode_reward += result.reward
                if done:
                    break

        return RunEpisodeResult(episode_number, episode_reward, self.hanabi_environment, \
            agent_step_times)
