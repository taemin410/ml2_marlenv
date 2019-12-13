
import gym
import envs
import argparse
import numpy as np
import random
import torch
import time
from PPO import ActorCritic

def gridexplore(args):
        
    env = gym.make('GridExplore-v0')

    env.reset()
    done_n = [False for _ in range(env.n_agents)]
    env.action_space[0].np_random.seed(123)
    totalreward = np.zeros(4)

    while not all(done_n):
        
        actions = []
        env.render()
        for i in range(env.n_agents):
            actions.append(env.action_space[i].sample())
        # actions[0] = int(input("move?"))
        # print(actions)
        s, r, done_n, _ = env.step(actions)

        totalreward  = totalreward + r 
        time.sleep(0.05)
    
    print("REWARDS: " , totalreward)
    env.render()

    env.close()

def test(args):

    env = gym.make('GridExplore-v0')
    device = torch.device('cpu')
    done_n = [False for _ in range(env.n_agents)]

    state_dim = env.observation_space[0].shape[0]

    action_dim = 5
    n_latent_var = 64           # number of variables in hidden layer
    model = ActorCritic(state_dim, action_dim, n_latent_var).to(device)


    path = r"./PPO_NOTSOLVED_GridExplore-v0.pth"
        
    model.load_state_dict(torch.load(path))
    print(model)
    state = env.reset()

    while not all(done_n):
        actions = []
        env.render()
        
        print(state.shape)
        outputs = model(state)

        a = torch.argmax(outputs, dim=1)
        actions[0]=a 

        s, r, done_n, _ = env.step(actions)

        totalreward  = totalreward + r 
        time.sleep(0.05)
    
    print("REWARDS: " , totalreward)
    env.render()

    env.close()




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML2-MULTIAGENT ENVS")

    parser.add_argument("--mode", type=str, default='single')
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)


