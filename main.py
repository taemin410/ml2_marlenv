
import gym
import envs
import argparse
import numpy as np
import random
import torch
import time
from PPO import Memory, ActorCritic, ConvNet, PPO

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
    acmodel = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
    
    model = ConvNet(action_dim)
    
    filename = "PPO_{}.pth".format(env_name)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(filename))

    acmodel.load_state_dict(torch.load(path))
    memory = Memory() 

    s = env.reset()
    totalreward = 0 

    while not all(done_n):
        actions = []
        env.render()
        
        state=np.array([s])
        
        state = torch.from_numpy(state).float()
        outputs = model(state)

        action = acmodel.act(outputs, memory)
        state, r, done_n, _ = env.step([action])

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


