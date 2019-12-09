import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

from multiagent.multi_discrete import MultiDiscrete
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace


class MultiAgentEnv(gym.Env):

    def __init__(self, n_agents, full_observation):


        self.n_agents = n_agents
        self.full_observation = full_observation

        self._obs_low =  0
        self._obs_high = 2

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._colaboration_reward = None
        self._step_count = 0 




    def reset(self):

        raise NotImplementedError


    def step(self, agents_actions):
        
        raise NotImplementedError




        # self._step_count += 1
        # rewards = [self._step_cost for _ in range(self.n_agents)]

        # for agent_i, action in enumerate(agents_actions):
        #     if not (self._agent_dones[agent_i]):
        #         reward[agent_i] = policy[agent_i](action)

        # for agent_i in range(self.n_agents):
        #     rewards[agent_i] += _reward

        # # if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
        # #     for i in range(self.n_agents):
        # #         self._agent_dones[i] = True

        # for i in range(self.n_agents):
        #     self._colaboration_reward[i] += rewards[i]

        # return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}


    # def get_agent_obs(self):
    #     _obs = []
    #     for agent_i in range(self.n_agents):
    #         pos = self.agent_pos[agent_i]
    #         _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates

    #         # check if prey is in the view area
    #         _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
    #         for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
    #             for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
    #                 if PRE_IDS['prey'] in self._full_obs[row][col]:
    #                     _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.

    #         _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
    #         _agent_i_obs += [self._step_count / self._max_steps]  # adding time
    #         _obs.append(_agent_i_obs)

    #     if self.full_observable:
    #         _obs = np.array(_obs).flatten().tolist()
    #         _obs = [_obs for _ in range(self.n_agents)]
    #     return _obs
