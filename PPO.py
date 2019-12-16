import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import envs 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter
import torchvision


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def __str__(self):
        return "{}\n{}\n{}\n{}\n{}".format(self.actions, self.states, self.logprobs, self.rewards, self.is_teminals)

    def randomsample(self, samplesize=16):
        sample={}
        sample["actions"] = self.actions[random]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)

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
        
    def forward(self, state):
        # TODO: forward function to run test

        # state = torch.from_numpy(state).float()
        
        # state_value = self.value_layer(state)

        
    def act(self, state, memory):
        # print(state)
        # state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
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
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def compute_advantage(self, values, batch, batch_mask):
        batch_size = len(batch)

        v_target = torch.FloatTensor(batch_size)
        advantages = torch.FloatTensor(batch_size)

        prev_v_target = 0
        prev_v = 0
        prev_A = 0

        for i in reversed(range(batch_size)):
            v_target[i] = batch[i] + GAMMA * prev_v_target * batch_mask[i]
            delta = batch[i] + GAMMA * prev_v * batch_mask[i] - values.data[i]
            advantages[i] = delta + GAMMA * TAU * prev_A * batch_mask[i]

            prev_v_target = v_target[i]
            prev_v = values.data[i]
            prev_A = advantages[i]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages, v_target
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        
        # TODO: MINI Batch sampling 
        m_a = memory.actions 
        m_s = memory.states 
        m_lp= memory.logprobs 
        m_r = memory.rewards 
        m_d = memory.is_terminals 
        
        indexes = torch.randperm(len(rewards))

        minibatch = 100 

        for start in range(0, len(rewards), minibatch):

            end = start + minibatch
            minibatch_idx = indexes[start:end]
            mini_batch = {}
            for k , v in 
            

        print(len(rewards))
        print(discounted_reward)


        # value + reward ->  value estimation 
        # Episode reduce -> batch -> minibatch 
        # Timestep , PPO1 AC env parellel action / Sampling in action 

        # Normalizing the rewards:
        # print(rewards)
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.flatten().float()

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        



        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            
          
            
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class ConvNet(nn.Module):
    def __init__(self, n_actions=5):
        super(ConvNet, self).__init__()
  
        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
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

        self.linear1 = nn.Linear(64 * 3 * 3, 512)

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
    max_episodes = 50000        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    writer = SummaryWriter("PPO on GridExplore-v0")


    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        
        # print("length of state arr is : " ,type(state))
        for t in range(max_timesteps):
            timestep += 1
           # env.render()

            state = np.array([state])

            img = torch.from_numpy(state).float().to(device)
            outputs = model(img)

            # Running policy_old:
            action = ppo.policy_old.act(outputs, memory)
            state, reward, done, _ = env.step([action])

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done[0])
            
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

#summary writer - > event add scalar -> stepnum 

if __name__ == '__main__':
    main()


