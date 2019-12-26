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

env.render()

env.close()
