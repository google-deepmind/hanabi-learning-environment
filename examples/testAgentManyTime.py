from __future__ import print_function

import numpy as np
from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.chenyang_agent import ChenyangAgent

# config du jeu
config = {"players": 2, "random_start_player": True}

#
game = rl_env.HanabiEnv(config)

#
agents = [SimpleAgent(config), SimpleAgent(config)]
res = 0
for i in range(0, 1000):
    # reset game
    obs = game.reset()
    done = False

    while not done:
        current_player = obs['current_player']

        actions = [agents[ind].act(obs_p1) for ind, obs_p1 in enumerate(obs['player_observations'])]

        # displaye current game action

        player = obs['current_player']
        obs, rew, done, info = game.step(actions[current_player])

        if done:
            # print(obs)

            result = sum(obs['player_observations'][current_player]['fireworks'].values())
            if obs['player_observations'][0]['life_tokens'] == 0:
                result = 0
            res += result
res = res / 1000
print("le score final sur 1000 paties est : ", res)
