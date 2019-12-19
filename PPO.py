import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import envs 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter
import torchvision
from random import random, sample
from collections import namedtuple


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.advantages= []
        
    def clear_memory(self):
        self.actions = []
        self.states = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.advantages= []

    def __str__(self):
        return "{}\n{}\n{}\n{}\n{}\n{}".format(self.actions, self.states,self.values, self.logprobs, self.rewards, self.is_teminals)

    def randomsample(self, batch_size=16):
        indexes= sample(self.mem, batch_size)
        return indexes

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        self.enc = ConvNet(action_dim).to(device)

        self.critic = nn.Sequential(
            nn.Linear(action_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
        )

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(action_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(action_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    # def forward(self, state):

    #     # TODO: forward function to run test

    #     state = torch.from_numpy(state).float()
    #     # out = F.relu(self.affine(state))

    #     # action_probs = self.action_layer(state)
        
    #     action_prob = F.softmax(self.action_layer(state), dim=-1)

    #     #state_value = self.value_layer(out)

    #     state_value = self.value_layer(state) 
        
    #     return action_prob , state_value
        
    def act(self, state, memory):
        # print(state)
        # state = torch.from_numpy(state).float().to(device) 
        outputs = self.enc(state)

        action_probs = self.action_layer(outputs)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(outputs)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, mini_batch_size = 32):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        

    
    def update(self, memory):   

        assert len(memory.states) == 2400 
        shuffleidx = np.random.permutation(len(memory.states))
        mini_batch_size = self.mini_batch_size

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        memory.advantages = rewards

        for _ in range(self.K_epochs):
            
            # convert list to tensor
            old_states = torch.stack(memory.states).to(device).detach()
            old_actions = torch.stack(memory.actions).to(device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(device).detach()
            
            for i in range(0, len(shuffleidx), mini_batch_size):
                
                end = i + mini_batch_size
                idx = np.array(shuffleidx[i:end])
                
                
                mb_s = [memory.states[i] for i in idx]
                mb_a = [memory.actions[i] for i in idx]
                mb_lp = [memory.logprobs[i] for i in idx]

                mb_adv = [torch.tensor(memory.advantages[i]) for i in idx]

                advs = torch.squeeze(torch.stack(mb_adv).to(device)).detach()

                mb_s = torch.squeeze(torch.stack(mb_s).to(device)).detach()
                mb_a = torch.squeeze(torch.stack(mb_a).to(device)).detach()
                mb_lp = torch.squeeze(torch.stack(mb_lp).to(device)).detach()

                rets =  mb_lp + advs

                advantages = torch.tensor(mb_adv).to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                advantages = advantages.flatten().float()

                # Evaluating old actions and values :
                logprobs, values, entropy = self.policy.evaluate(mb_s, mb_a)
                ent = entropy.mean()
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - mb_lp.detach())

                # vf_loss = 0.5*self.MseLoss(rets , values) 

                # Finding Surrogate Loss:
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(advs , values) - 0.01*ent

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        
        self.policy_old.load_state_dict(self.policy.state_dict())

        # # value + reward ->  value estimation 
        # # Episode reduce -> batch -> minibatch 
        # # Timestep , PPO1 AC env parellel action / Sampling in action 

class ConvNet(nn.Module):
    def __init__(self, n_actions=5, state_dim=4):
        super(ConvNet, self).__init__()
  
        self.conv1 = nn.Conv2d(state_dim, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.enc = nn.Sequential(
            self.conv1, 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2, 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # self.conv3, 
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #64 *3 * 3
        self.linear1 = nn.Linear(256, 512)

        self.fc = nn.Linear(512, n_actions)
        
    def forward(self, x):
        out = self.enc(x)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.fc(out)
        return out


def main():
    ############## Hyperparameters ##############
    env_name = "GridExplore-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space[0].shape[0]

    action_dim = 5
    model = ConvNet(action_dim).to(device)

    render = False
    solved_reward = 50         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2400      # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 2                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    mini_batch_size = 32
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    buffer = {key:value for key,value in memory.__dict__.items() if not key.startswith('__') and not callable(key)}

    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    writer = SummaryWriter("logs")


    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        
        # print("length of state arr is : " ,type(state))
        for t in range(max_timesteps):
            timestep += 1
           # env.render()

            state = np.array([state])

            outputs = torch.from_numpy(state).float().to(device)

            # Running policy_old:
            action = ppo.policy_old.act(outputs, memory)
            state, reward, done, _ = env.step([action])

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.dones.append(done[0])
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward[0]
            if render:
                env.render()
            if all(done):
                break
                
        avg_length += t
        
        writer.add_scalar('step/running_reward', running_reward, timestep)
        
        grid = torchvision.utils.make_grid(torch.tensor(env.grid))
        writer.add_image('images', grid, max_timesteps)


        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    writer.close()
    torch.save(ppo.policy.state_dict(), './PPO_NOTSOLVED_{}.pth'.format(env_name))


if __name__ == '__main__':
    main()


