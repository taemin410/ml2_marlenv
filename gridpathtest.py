import gym
import envs

env = gym.make('GridPath-v0')
env.render()
env.reset()
done = [False for i in range(env.n_agents)]
while not all(done):

	done = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])

	env.render_graphic()

env.render_graphic()

env.close()
