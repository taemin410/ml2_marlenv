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



inputs = ["11", "22", "33", "44", "11", "22", "33", "00", "44", "11", "22", "33", "44"]


def test_Checkers_input():
    env = gym.make("Checkers-v0")

    for ep_i in range(len(inputs)):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        obs_n = env.reset()
        env.render()
        while not all(done_n):
            action_n = [int(_) for _ in inputs[ep_i]]
            print(action_n)

            obs_n, reward_n, done_n, _ = env.step(action_n)
            ep_reward += sum(reward_n)
            env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()
    

def test_Snake():


	env = gym.make('Snakegame-v0')

	env.reset()
	done_n = [False for _ in range(env.n_agents)]

	while not all(done_n):
	    
	    actions = []
	    env.render()
	    for i in range(env.n_agents):
	        actions.append(env.action_space[i].sample())
	        
	    s, r, done_n, _ = env.step(actions)
	    
	    time.sleep(0.25)

	env.close()

