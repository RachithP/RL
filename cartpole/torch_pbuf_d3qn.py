# -*- coding: utf-8 -*-
import random, sys
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.PriorityBuffer import *

EPISODES = 1000                 # add these kinda variable to a config file and import for better code structure

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_1 = nn.Linear(self.state_size, 24)
        self.hidden_2 = nn.Linear(24, 24)
        self.hidden_3 = nn.Linear(24, 24)
        self.output_1 = nn.Linear(24, 1)
        self.output_2 = nn.Linear(24, self.action_size)


    def forward(self, state):
        x = F.relu(self.hidden_1(state))
        x1 = F.relu(self.hidden_2(x))
        x2 = F.relu(self.hidden_3(x))
        y1 = self.output_1(x1)
        y2 = self.output_2(x2)

        output = y1 + (y2 - y2.mean(dim=1, keepdim=True))

        return output

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.max_memory_size = 2000
        self.memory = PriorityBuffer(max_size=self.max_memory_size)
        self.batch_size = 32
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.1
        self.model = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mseloss = nn.MSELoss()
        self.max_num_param_update = 100
        self.num_param_update = 0

    def run_agent_randomly(self, env):
        """
            Run this function to fill the memory with random initial values.
        """
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])

        for _ in range(self.max_memory_size):
            
            action = random.randrange(self.action_size)

            next_state, reward, done, _  = env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            self.memory.store((state, action, reward, next_state, done))

            if done:
                state = env.reset()
                state = np.reshape(state, [1, self.state_size])
            else:
                state = next_state

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            act_values = self.model.forward(state).squeeze()
            return act_values.max(0)[1].item()  # returns action

    def update(self):
        
        minibatch_experiences, minibatch_IS_weights, minibatch_tree_indices = self.memory.sample(self.batch_size)

        minibatch_experiences = np.reshape(
            minibatch_experiences, (self.batch_size, len(minibatch_experiences[0])))

        states = torch.FloatTensor(list(minibatch_experiences[:, 0])).reshape(
            self.batch_size, -1).to(self.device)
        actions = torch.LongTensor(list(minibatch_experiences[:, 1])).reshape(
            self.batch_size, -1).to(self.device)
        rewards = torch.FloatTensor(list(minibatch_experiences[:, 2])).reshape(
            self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(list(minibatch_experiences[:, 3])).reshape(
                self.batch_size, -1).to(self.device)
        dones = torch.FloatTensor(list(minibatch_experiences[:, 4])).reshape(
            self.batch_size, -1).to(self.device)
        weights = torch.FloatTensor(minibatch_IS_weights).reshape(
            self.batch_size, -1).to(self.device).squeeze()

        # Get indicies of actions with max value according to online model
        model_actions = self.model.forward(next_states).detach().max(1)[1]

        # gather Q values of states and above actions from target model
        target_q_values = self.target_model.forward(next_states).detach().gather(1, model_actions.unsqueeze(1)).squeeze()

        targets = rewards.squeeze() + self.gamma * target_q_values *(1-dones).squeeze()

        # get q-values corresponding to actions at that step
        state_action_values = self.model.forward(
                states).gather(1, actions).squeeze()

        td_errors = targets - state_action_values

        self.memory.update_priorities_batch(minibatch_tree_indices, np.abs(td_errors.cpu().detach().numpy()))

        loss = (weights*(td_errors)**2).mean()
     
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_param_update += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    agent = DQNAgent(state_size, action_size)
    
    agent.run_agent_randomly(env)
    # agent.load("./save/cartpole-dqn.h5")
    
    done = False
    average_reward = 0

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

            # if(agent.num_param_update%agent.max_num_param_update==0):
            #     agent.hard_target_network_update()
            agent.update()

            if(time % 4 == 0):
                agent.soft_target_network_update()


            if done:
                agent.score_list[e] = time
                if(e>98):
                    average_reward = np.sum(agent.score_list[e-99:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, agent.score_list[e], agent.epsilon, average_reward))
                break
            
    #agent.save("cartpole-dqn.h5")
    #np.savetxt("dqn_scores.txt",agent.score_list,fmt='%.8f')

    plt.figure(1)
    #plt.subplot(121)
    plt.plot(agent.score_list, label="episodic score")
    #plt.subplot(122)
    #plt.plot(average_reward, label="average score from all episodes")
    #plt.legend()
    plt.show()

    env.close()