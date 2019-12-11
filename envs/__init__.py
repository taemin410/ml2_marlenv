from gym.envs.registration import registry, register, make, spec
import gym
import os 
import numpy as np
import random 

from envs.snakegame.common import Direction, Point
from envs.snakegame.python import Python 

register(
	id='Pong-2p-v0',
	entry_point='envs.pong:PongGame',
)


register(
	id='Checkers-v0',
	entry_point='envs.checkers:Checkers',
)

register(
	id='PongDuel-v0',
	entry_point='envs.pong_duel:PongDuel',
)

#    kwargs={'full_observable': True}



register(
	id='PredatorPrey-v0',
	entry_point='envs.predator_prey:PredatorPrey'
	)

register(
	id='PredatorPrey-v1',
	entry_point='envs.predator_prey:PredatorPrey',
	kwargs={
		'full_observable' : True
	
		}
	)

register(
    id='Combat-v0',
    entry_point='envs.combat:Combat',
)


register(
	id='Combat-v1',
    entry_point='envs.combat:Combat',
	kwargs={
		'full_observable' : True
	
		}
	)

init_map = os.path.join('envs/snakegame' ,'10x10.txt')
#randomize snake's initial position 
players = [
    Python(Point(np.random.randint(3, 7, 1)[0],np.random.randint(3, 7, 1)[0]), Direction.SOUTH, 1)

]

register(
	id='Snakegame-v0',
    entry_point='envs.snakegame:SnakeGameMultiEnv',
	kwargs={
		'full_observable' : True , 
		'init_map' : init_map,
		'players' : players		
		}
	)

register(
	id='GridExplore-v0',
	entry_point='envs.grid_explore:GridExplore',
	kwargs={
		'full_observable' : False,
		'size' : 15
	}
)

