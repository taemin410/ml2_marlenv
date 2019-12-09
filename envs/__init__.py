from gym.envs.registration import registry, register, make, spec
import gym


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
