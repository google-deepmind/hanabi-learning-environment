import sys
import os
import getopt


currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from hanabi_learning_environment.rl_env import HanabiEnv

from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent


def get_hanabi_env()->HanabiEnv:

    AGENT_CLASSES = {'HTGSAgent' : RandomAgent}

    flags = {'players': 5, 'num_episodes': 100, 'agent_class': 'RandomAgent'}

    options, arguments = getopt.getopt(sys.argv[1:], '',
                                    ['players=',
                                    'num_episodes=',
                                    'agent_class='])
    if arguments:
        sys.exit('usage: rl_env_example.py [options]\n'
                '--players       number of players in the game.\n'
                '--num_episodes  number of game episodes to run.\n'
                '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))

    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)
        
    environment = rl_env.make('Hanabi-Full', num_players=flags['players'], agentObservationType = pyhanabi.AgentObservationType.CARD_KNOWLEDGE)


    observations = environment.reset()

    return observations

def main():
    observation: HanabiEnv = get_hanabi_env()
    print(observation)

if __name__ == "__main__":
    main()