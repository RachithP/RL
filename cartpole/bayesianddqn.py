# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import pymc3 as pymc

EPISODES = 1000                 # add these kinda variable to a config file and import for better code structuring

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, feature_size):
        super(DQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, self.feature_size),
            nn.ReLU()
            # nn.Linear(24, self.action_size)  # Removing output-layer and replacing it with BLR layer
        )

    def forward(self, state):
        return self.model(state)

class DQNAgent:
    def __init__(self, state_size, action_size, feature_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size
        self.memory = deque(maxlen=2000)   
        self.batch_size = 32
        self.min_mem_for_replay = self.batch_size * 4
        self.gamma = 0.99    # discount rate
        self.learning_rate = 0.001
        self.tau = 0.1
        self.model = DQNModel(self.state_size, self.action_size, self.feature_size).to(self.device)
        self.target_model = DQNModel(self.state_size, self.action_size, self.feature_size).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mseloss = nn.MSELoss()
        self.max_num_param_update = 99
        self.num_param_update = 0
        # Baeyesian regression required parameters
        self.sigma_w = 0.001      # W prior variance. This is assuming equal variance and co-variance=0 in all dimension of w.
        self.sigma_n = 0.1          # noise variance
        self.initialize_bayesian_matrices()
        self.f_sampling = 100   # frequency of thompson sampling
        self.bayes_batch_size = 100

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return torch.mm(self.mean_W_, self.model.forward(state).transpose(0, 1)).max(0)[1].item()

    def update(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        minibatch = np.reshape(minibatch, (len(minibatch), len(minibatch[0])))

        states = torch.FloatTensor(list(minibatch[:, 0])).reshape(len(minibatch), -1).to(self.device)
        actions = torch.LongTensor(list(minibatch[:, 1])).reshape(len(minibatch), -1).to(self.device)
        rewards = torch.FloatTensor(list(minibatch[:, 2])).reshape(len(minibatch), -1).to(self.device)
        next_states = torch.FloatTensor(list(minibatch[:, 3])).reshape(len(minibatch), -1).to(self.device)
        dones = torch.FloatTensor(list(minibatch[:, 4])).reshape(len(minibatch), -1).to(self.device)

        # Get indicies of actions with max value according to online model
        model_actions = torch.mm(self.model.forward(next_states).detach(), self.mean_W_.transpose(0, 1)).max(1)[1]
        
        # gather Q values of states and above actions from target model
        target_q_values =  torch.mm(self.target_model.forward(next_states).detach(), self.mean_W_target.transpose(0, 1)).gather(1, model_actions.unsqueeze(1)).squeeze()

        targets = rewards.squeeze() + self.gamma * target_q_values *(1-dones).squeeze()

        # get q-values corresponding to actions at that step
        state_action_values = torch.mm(self.model.forward(states), self.mean_W.transpose(0, 1)).gather(1, actions).squeeze()

        loss = self.mseloss(state_action_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_param_update += 1

    def initialize_bayesian_matrices(self):
        self.identity_feature_size = torch.eye(self.feature_size, dtype=torch.float32).to(self.device)
        self.mean_W = torch.empty(size=(self.action_size, self.feature_size)).normal_(mean=0, std=.01).to(self.device)
        self.mean_W_target = torch.empty(size=(self.action_size, self.feature_size)).normal_(mean=0, std=.01).to(self.device)
        self.mean_W_ = torch.empty(size=(self.action_size, self.feature_size)).normal_(mean=0, std=.01).to(self.device)
        self.cov_W = torch.empty(size=(self.action_size, self.feature_size, self.feature_size)).normal_(mean=0, std=1.0).to(self.device) \
            + self.identity_feature_size
        self.cov_W_decom = self.cov_W.data

        for i in range(self.action_size):
            self.cov_W[i] = self.identity_feature_size.data
            self.cov_W_decom[i] = torch.cholesky(((self.cov_W[i]+torch.transpose(self.cov_W[i], 0, 1))/2.))
        self.cov_W_target = self.cov_W.data

    def bayes_reg(self):

        #  as given in paper, initialized to zero. Else, need to do moment version taking into account previous values.
        phiphiT = torch.zeros((self.action_size, self.feature_size, self.feature_size), dtype=torch.float32).to(self.device)
        phiY = torch.zeros((self.action_size, self.feature_size), dtype=torch.float32).to(self.device)

        if self.bayes_batch_size > len(self.memory):
            minibatch = random.sample(self.memory, len(self.memory))
        else:
            minibatch = random.sample(self.memory, self.bayes_batch_size)

        minibatch = np.reshape(minibatch, (len(minibatch), len(minibatch[0])))

        states = torch.FloatTensor(list(minibatch[:, 0])).reshape(len(minibatch), -1).to(self.device)
        actions = torch.LongTensor(list(minibatch[:, 1])).reshape(len(minibatch), -1).to(self.device)
        rewards = torch.FloatTensor(list(minibatch[:, 2])).reshape(len(minibatch), -1).to(self.device)
        next_states = torch.FloatTensor(list(minibatch[:, 3])).reshape(len(minibatch), -1).to(self.device)
        dones = torch.FloatTensor(list(minibatch[:, 4])).reshape(len(minibatch), -1).to(self.device)

        # filter by actions and update posterior for weights corresponding to each action
        for i in range(self.action_size):
            ind = (actions == i).squeeze()
            mini_states = states[ind, :]
            mini_rewards = rewards[ind, :]
            mini_next_states = next_states[ind, :]
            mini_dones = dones[ind, :]

            phi_a               = self.model.forward(mini_states).detach().squeeze()
            phiphiT[i]          = torch.mm(phi_a.transpose(0, 1), phi_a)
            try:
                target_fn_q_values  = torch.max(torch.mm(self.target_model.forward(mini_next_states).detach(), self.mean_W_target.transpose(0, 1)), dim=1)[0]
            except:
                print(torch.mm(self.target_model.forward(mini_next_states).detach(), self.mean_W_target.transpose(0, 1)))
                quit()
            target_bellman      = mini_rewards.squeeze() + self.gamma * target_fn_q_values *(1-mini_dones).squeeze()
            phiY[i]             = torch.mm(phi_a.transpose(0, 1), target_bellman.unsqueeze(1)).squeeze()
            inv                 = torch.inverse((phiphiT[i] / self.sigma_n) + (self.identity_feature_size / self.sigma_w))

            self.mean_W[i] = torch.mm(inv, phiY[i].unsqueeze(1)).squeeze() / self.sigma_n
            self.cov_W[i] = self.sigma_w * inv
            self.cov_W_decom[i] = torch.cholesky((self.cov_W[i]+self.cov_W[i].transpose(0, 1))/2.)

        self.mean_W_target = self.mean_W.data
        self.cov_W_target  = self.cov_W.data

    def thompson_sampling(self):
        for i in range(self.action_size):
            val = torch.empty(size=(self.feature_size, 1)).normal_(mean=0, std=1).to(self.device)
            self.mean_W_[i] = torch.mm(self.cov_W_decom[i], val).squeeze() + self.mean_W[i]

    def soft_target_network_update(self):
        for q_weight, target_weight in zip(self.model.parameters(), self.target_model.parameters()):
            target_weight.data.copy_(self.tau * q_weight + (1-self.tau)*target_weight)

    def hard_target_network_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, feature_size=24)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    average_reward = 0
    ct = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            #if(e>500):
            #    env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # thompson sampling to nudge the distribution on weights
            if time%agent.f_sampling==0:
                agent.thompson_sampling()

            # train the model
            if len(agent.memory) > agent.min_mem_for_replay:
                agent.update()
            
            # update target model
            # if(agent.num_param_update%agent.max_num_param_update==0):
            #     agent.hard_target_network_update()
                # ct += 1
                if(time % 4 == 0):
                    agent.soft_target_network_update()

                    #  update posterior on  weight
                    # if ct%10==0:
                    agent.bayes_reg()
                    # ct = 0

            if done:
                agent.score_list[e] = time
                if(e>98):
                    average_reward = np.sum(agent.score_list[e-99:])/100.0
                print("episode: {}/{}, score: {}, average_reward: {}"
                        .format(e, EPISODES, time, average_reward))
                break
            
    #agent.save("cartpole-dqn.h5")
    #np.savetxt("dqn_scores.txt",agent.score_list,fmt='%.8f')

    plt.figure(1)
    #plt.subplot(121)
    plt.plot(agent.score_list, label="episodic score")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    #plt.subplot(122)
    #plt.plot(average_reward, label="average score from all episodes")
    #plt.legend()
    plt.show()

    env.close()