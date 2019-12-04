import gym
from gym.envs.registration import registry, register, make, spec
import envs






env = gym.make('PredatorPrey-v0')
env.reset()
done_n = [False for _ in range(env.n_agents)]
import time

while not all(done_n):
	
	env.render()
	s, r, done_n, _ = env.step(env.action_space.sample())
	ep_reward = sum(r)

	print(s,r)
	print(done_n)
	print(_)
	time.sleep(1)

env.close()

