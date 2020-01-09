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
from PIL import Image
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
import tensorflow as tf 
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm

from stable_baselines import DQN, PPO2, A2C
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, ActorCriticPolicy, CnnPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.deepq.policies import FeedForwardPolicy as DqnFFPolicy


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="cnn", cnn_extractor =custom_cnn )

class DqnCnnPolicy(DqnFFPolicy):
    def __init__(self, *args, **kwargs):
        super(DqnCnnPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="cnn", cnn_extractor =custom_cnn )

# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)
register_policy('DqnCnnPolicy', DqnCnnPolicy)


def train():

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

    
def random():

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


def test():


        env = gym.make("GridExplore-v0")

        model = PPO2.load("ppo2_GridExplore.pth")
        images = []
        obs = env.reset()
        done = [False for i in range(4)] 
        while not all(done):
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)

                images.append(env.render_graphic())
                time.sleep(0.05)
        
        images.append(env.render_graphic())

        env.close()

        images[0].save('out.gif', save_all=True, append_images=images)


