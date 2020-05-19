import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#=============================================================================#
class Multi_Agent(nn.Module):    
    def __init__(self, state_size, action_size, num_agents=2, GAMMA=0.99, TAU=1e-3, EPS_S=1, EPS_E=0.01, EPS_D=0.997, BUF_S=1e6, BCH_S=128, LR_A=1e-3, LR_C=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents=num_agents
        self.gamma = GAMMA
        self.tau = TAU
        self.epsilon_start = EPS_S
        self.epsilon_end = EPS_E
        self.epsilon_decay = EPS_D
        self.replay_buffer_size = BUF_S
        self.batch_size = BCH_S
        self.lr_actor = LR_A
        self.lr_critic = LR_C
    
        self.agents=[Agent(self.state_size,\
                           self.action_size,\
                           self.gamma,\
                           self.tau,\
                           self.epsilon_start,\
                           self.epsilon_decay,\
                           self.epsilon_end,\
                           self.replay_buffer_size,\
                           self.batch_size,\
                           self.lr_actor,\
                           self.lr_critic) for _ in range(self.num_agents)]

    def act(self, states, add_noise=True):
        actions = [self.agents[i].act(states) for i in range(self.num_agents)]
        actions = np.reshape(actions, (1, self.action_size*self.num_agents))
        return actions

    def step(self, states, actions, rewards, next_states, done):        
        for i in range(self.num_agents):
            self.agents[i].step(states, actions, rewards[i], next_states, done, i)
            
    def decay(self):
        for i in range(self.num_agents):
            self.agents[i].update_noise_scaling()
    
    def save(self):
        for i in range(self.num_agents):
            self.agents[i].save(i)
           
    def reset(self):
        for i in range(self.num_agents):
            self.agents[i].reset()

#=============================================================================#    
class Agent:
    def __init__(self, state_size, action_size, gamma=0.99, tau=1e-2,
                 epsilon_start=1.0, epsilon_decay=1.0, epsilon_end=1.0, replay_buffer_size=int(1e5), batch_size=128, lr_actor=5e-4, lr_critic=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 0
        self.GAMMA = gamma
        self.TAU = tau
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, self.seed).to(device)
        self.critic_target = Critic(state_size, action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0)

        # Noise process
        self.noise = OUNoise((1, self.action_size), self.seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.replay_buffer_size, self.batch_size, self.seed)


    def step(self, state, action, reward, next_state, done, i):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA, i)

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            actions += self.epsilon * self.noise.sample()
        return np.clip(actions, -1, 1)
  
    def save(self,i):
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
        torch.save(self.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(i))
        
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number=None):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)    # 128x2

        if agent_number is not None:
            if agent_number == 0:
                actions_next = torch.cat((actions_next, actions[:, 2:]), dim=1)
            else:
                actions_next = torch.cat((actions[:, :2], actions_next), dim=1)

        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        if agent_number is not None:
            if agent_number == 0:
                actions_pred = torch.cat((actions_pred, actions[:, 2:]), dim=1)
            else:
                actions_pred = torch.cat((actions[:, :2], actions_pred), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)
        
        
    def update_noise_scaling(self):
        # ---------------------------- update noise scaling ---------------------------- #
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.13, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.state = copy.copy(self.mu)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)