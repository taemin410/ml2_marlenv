import gym
from gym.envs.registration import registry, register, make, spec
import envs
import time





def test_PredatorPrey():

	env = gym.make('PredatorPrey-v0')
	env.reset()
	done_n = [False for _ in range(env.n_agents)]

	while not all(done_n):
		
		env.render()
		s, r, done_n, _ = env.step(env.action_space.sample())
		ep_reward = sum(r)

	env.close()

def test_Pong2p():


	env = gym.make('Pong-2p-v0')
	env.reset()
	done_n = False

	while not done_n:
		
		env.render()
		s, r, done_n, _ = env.step(env.action_space.sample())

	env.close()

def test_Combat():

	env = gym.make('Combat-v0')
	env.reset()
	done_n = [False for _ in range(env.n_agents)]

	while not all(done_n):
		
		env.render()
		s, r, done_n, _ = env.step(env.action_space.sample())
		
		time.sleep(0.05)

	env.close()



