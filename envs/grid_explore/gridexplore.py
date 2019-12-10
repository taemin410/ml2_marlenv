from gridworld import GridWorld 


import numpy as np


class Cell:
    UNVISITED = 0
    VISITIED = 1
    WALL = 2
    AGENTS = [3, 4, 5, 6]
    

class GridExplore(GridWorld):

	def __init__(self, size):

		self.size = size
		self.grid = [ [0 for _ in range(size)] for _ in range(size)]


		self.__setwall()
		self.render()


	def reset(self):

		return ""



	def step(self, actions):

		return actions


	def render(self):
		for i in self.grid:
			print(i)


	def __setwall(self):
		for i in range(self.size):
			self.grid[0][i] = 1 
			self.grid[self.size-1][i] = 1 
			self.grid[i][0] = 1
			self.grid[i][self.size-1]= 1


	def __getEmptyCell(self):
		

a = GridExplore(10)

