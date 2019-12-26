from gym.envs.registration import registry, register, make, spec
import gym
import os 
import numpy as np
import random 


register(
	id='GridExplore-v0',
	entry_point='envs.grid_explore:GridExplore',
	kwargs={
		'full_observable' : False,
		'size' : 8
	}
)


register(
	id='GridPath-v0',
	entry_point='envs.grid_explore:GridPath',
	kwargs={
		'full_observable' : False,
		'size' : 11
	}
)