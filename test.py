import gym
from gym.spaces import Discrete, Box
# from ray import tune
import time 
import envs
from envs.snakegame.graphics import ML2PythonGUI
import policy
import argparse
import torch 

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



def human_1p(args):
    env = gym.make('Snakegame-v0')

    n_ac = env.action_space[0].n
    in_shape = (env.observation_space[0].shape[1]*2,
                *env.observation_space[0].shape[2:])
    net = policy.PythonNet(in_shape, n_ac)

    gui = ML2PythonGUI(env, args)
    gui.run(net)

parser = argparse.ArgumentParser(description="I Won(Tae)-Chu!")

parser.add_argument("--tag", type=str, default='snake_test')
parser.add_argument("--mode", type=str, default='single')
parser.add_argument("--seed", type=int, default=100)

parser.add_argument_group("interface options")
parser.add_argument("--human", action='store_true')
parser.add_argument("--cell_size", type=int, default=20)

args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


human_1p(args)
