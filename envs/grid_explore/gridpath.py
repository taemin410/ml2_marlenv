from envs.grid_explore.gridworld import GridWorld 
import random
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

import numpy as np
import math
from gym import spaces
from PIL import ImageColor
import copy

CELL_SIZE = 30

PRE_IDS = {
    'agent': ['3','4','5','6'],
    'wall': '2',
    'empty': '0',
    'visited':'1'
}

WALL_COLOR = 'black'
VISITED_COLOR = 'grey'
AGENT_COLOR = 'green'

class Cell:
    UNVISITED = 0
    DESTINATIONS = 1
    WALL = 2
    AGENTS = [3, 4, 5, 6]

class Move:
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	STAY = 4 

class Rewards:
	TIMEPENALTY = -0.1
	EXPLORATION_BONUS = 1
	WIN = 100 

class Agent:

	def __init__(self, posx, posy, idx, sight=2):
		
		self.idx = int(idx)
		self.x = posx
		self.y = posy 
		self.position = (self.x, self.y)
		
		self.sight = sight

	def __str__(self):
		return "x: " + str(self.x) +  " y: "+ str(self.y)+ " agent idx: " + str(self.idx)

			
	def move(self, MOVE, grid):
		new_x, new_y = self.position[0], self.position[1]
		org_x, org_y = self.position[0], self.position[1]

		if MOVE==Move.UP and self.position[1] != 1:
			new_y -= 1 
		elif MOVE == Move.DOWN and self.position[1] != len(grid)-1:
			new_y += 1 	
		if MOVE==Move.LEFT and self.position[0] != 1:
			new_x -= 1 
		elif MOVE == Move.RIGHT and self.position[0] != len(grid)-1:
			new_x += 1 	

		#check if wall exist there
		if grid[new_y][new_x] != 2 and grid[new_y][new_x] not in Cell.AGENTS:
			self.makeMove(new_x, new_y)
			# grid[org_y][org_x] = 1
			return new_x, new_y
		else:
			return self.position[0], self.position[1]

	def makeMove(self, new_x, new_y):
		self.x = new_x
		self.y = new_y
		self.position = (self.x, self.y)


class GridPath(GridWorld):

	def __init__(self, size, n_agents=4, full_observable=False, dist_penalty=5):

		self.size = size
		#initialize with WALL
		self.grid = [ [Cell.WALL for _ in range(size)] for _ in range(size)]
		self._grid_shape=[size,size]

		self.agentList=[]
		self.time= 0
		self.n_agents=n_agents
		self.dist_penalty = dist_penalty 

		self.__setpath()
		self.__setDestination()
		self.__setStartingPosition()

		self.init_agent_pos= {}
		self.viewer = None

		self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0,high=6,shape=(4, self.size, self.size)) for _ in range(self.n_agents)])
	
		self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])

		self.render()

	def render(self):
		for i in self.grid:
			print(i)
		print("")
	
	def __setpath(self):
		for i in range(1, self.size-1):
			self.grid[self.size-2][i] = 0
			self.grid[self.size-3][i] = 0
			self.grid[1][i] =0
			self.grid[2][i] =0

			#Path generation
			self.grid[i][self.size//2] = 0

	def __setDestination(self):
		self.grid[1][1] = self.grid[1][self.size-2] = 1
		self.grid[2][1] = self.grid[2][self.size-2] = 1

	#set starting position for 4 players
	def __setStartingPosition(self):
		assert self.n_agents == 4
		self.grid[self.size-2][1] = 3
		self.grid[self.size-3][1] = 4
		self.grid[self.size-2][self.size-2] = 5
		self.grid[self.size-3][self.size-2] = 6

	

	def reset(self):
		self.grid = [ [Cell.WALL for _ in range(size)] for _ in range(size)]
		self.__setpath()
		self.__setDestination()
		self.__setStartingPosition()

		self.dones = np.zeros(self.n_agents, dtype=bool)

		
	def step(self, actions):
		return actions




	def observation(self):

		statearray= np.zeros(self.observation_space[0].shape) 
		for i in self.agentList:

			state = np.zeros(self.observation_space[0].shape)

			agents = np.isin(self.grid, Cell.AGENTS ).astype(np.float32)	
			agenti = np.isin(self.grid, i.idx ).astype(np.float32)			

			visited = np.isin(self.grid, Cell.VISITED).astype(np.float32) 
			#add agent's position as visited
			visited = visited + agents
			wall = np.isin(self.grid, Cell.WALL).astype(np.float32)

			#exclude self from agents pos list
			agents = agents - agenti

			for idx in range(self.n_agents):
				state[0] = agenti
				state[1] = agents
				state[2] = visited
				state[3] = wall

				statearray = state
		
		return statearray