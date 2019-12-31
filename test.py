import gym
from gym.spaces import Discrete, Box
# from ray import tune
import time 
import envs
# from envs.snakegame.graphics import ML2PythonGUI
import policy
import argparse
import torch 
import numpy as np
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines import PPO2

# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         return env

#     return _init

# env_id = "GridExplore-v0"
# num_cpu = 2 # Number of processes to use
# # Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

# model= PPO2(MlpPolicy, env, verbose =1 )

# class SimpleCorridor(gym.Env):
#     def __init__(self, config):
#         self.end_pos = config["corridor_length"]
#         self.cur_pos = 0
#         self.action_space = Discrete(2)
#         self.observation_space = Box(0.0, self.end_pos, shape=(1, ))

#     def reset(self):
#         self.cur_pos = 0
#         return [self.cur_pos]

#     def step(self, action):
#         if action == 0 and self.cur_pos > 0:
#             self.cur_pos -= 1
#         elif action == 1:
#             self.cur_pos += 1
#         done = self.cur_pos >= self.end_pos
#         return [self.cur_pos], 1 if done else 0, done, {}



# tune.run(
#     "PPO",
#     config={
#         "env": SimpleCorridor,
#         "num_workers": 3,
#         "env_config": {"corridor_length": 5}})



# def human_1p(args):
#     env = gym.make('Snakegame-v0')

#     n_ac = env.action_space[0].n
#     in_shape = (env.observation_space[0].shape[1]*2,
#                 *env.observation_space[0].shape[2:])
#     net = policy.PythonNet(in_shape, n_ac)

#     gui = ML2PythonGUI(env, args)
#     gui.run(net)

# parser = argparse.ArgumentParser(description="I Won(Tae)-Chu!")

# parser.add_argument("--tag", type=str, default='snake_test')
# parser.add_argument("--mode", type=str, default='single')
# parser.add_argument("--seed", type=int, default=100)

# parser.add_argument_group("interface options")
# parser.add_argument("--human", action='store_true')
# parser.add_argument("--cell_size", type=int, default=20)

# args = parser.parse_args()
# args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# human_1p(args)


env = gym.make('GridExplore-v1')


env.reset()
done_n = [False for _ in range(env.n_agents)]
env.action_space[0].np_random.seed(123)
totalr= [0 for _ in range(env.n_agents)] 
while not all(done_n):
    
    actions = []
    env.render()
    for i in range(env.n_agents):
        actions.append(env.action_space[i].sample())
    s, r, done_n, _ = env.step(actions)
    print("REWARDS: " , r)
    totalr += r
    
    time.sleep(0.05)

print("TOTAL REWARDS: " , totalr)
env.render()

env.close()


# print( a.isNear(a.agentList[0], a.agentList[2],5 ))

