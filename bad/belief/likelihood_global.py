import sys
import os


from bad.encoding.observation import Observation
from likelihood_player import LikelihoodPlayer

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from action_network import ActionNetwork
from hanabi_learning_environment.rl_env import HanabiEnv

class Likelihood(list):
    def __init__(self, constants, action_network: ActionNetwork, old_hint_matrix = None,
                observation: Observation = None, last_act=None, old_likelihood: 'Likelihood'= None)-> None:
        """Initialize / Update the likelihood based on the last_action, observation and last_action network
        By initializing the likelihood, the first time last_act,old_likelihood and last_act are None

        Args:
            constants (HanabiEnv): Hanabi environment 
            observation (Observation): Part of Input from Network
            act_network (ActionNetwork): Network that predicts the last_action
            last_act (_type_, None): the last action taken by the agent
            old_likelihood (Likelihood, None): Likelikhood from the previous step aka current likelihood
            old_hint_matrix (HintMatrix, None): Hint matrix from the previous step
        Returns:
            likelihood (Likelihood): Likelihood of the current step
        """
        super().__init__(self.__init(constants, observation, 
                                     action_network, last_act, old_likelihood, old_hint_matrix))


    def __init(self, constants, observation: Observation,
                     action_network: ActionNetwork, last_act, 
                     old_likelihood: 'Likelihood', old_hint_matrix) -> list:
        
        players_hands = [LikelihoodPlayer(constants, idx_ply, observation, 
                         action_network, last_act, old_likelihood, old_hint_matrix) 
                         for idx_ply in range(constants.num_ply)] 

        return players_hands