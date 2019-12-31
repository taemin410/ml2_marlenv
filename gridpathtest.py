import gym
import envs
import argparse
import numpy as np
import random
import torch
import time
from PPO import Memory, ActorCritic, ConvNet


env = gym.make('GridPath-v0')
env.render()
env.reset()
done = [False for i in range(env.n_agents)]
while not all(done):

	done = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])

	env.render()

env.render()

env.close()
