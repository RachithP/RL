# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import Logger
from utils.PriorityBuffer import *

EPISODES = 1000

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def forward(self, state):
        return self.model(state)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.folder = "./graphs/DDQN_PER_SOFT"
        self.writer = SummaryWriter(log_dir=self.folder)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.max_memory_size = 100000
        self.memory = PriorityBuffer(max_size=self.max_memory_size)
        self.batch_size = 32
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.adam_learning_rate = 0.001
        self.rms_learning_rate = 0.0025
        self.rms_eps = 0.01
        self.rms_momentum = 0.9
        self.rms_alpha = 0.99
        self.tau = 0.1
        self.model = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.rms_learning_rate, eps=self.rms_eps, alpha=self.rms_alpha, 
        #  momentum=self.rms_momentum, centered=True)
        self.loss = nn.MSELoss()
        self.target_update_soft = True
        self.target_update_freq_hard = 100
        self.target_update_freq_soft = 10
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
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    Logger.Logger(agent)    # Log agent parameters in a file
    agent.run_agent_randomly(env)   # run agent to accumulate memory points
    average_reward = 0
    cnt = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        done = False
        ep_cnt = 0

        while not done:
            #if(e>500):
            #    env.render()
            ep_cnt += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward

            agent.update()
            agent.writer.add_scalar("Score per model update", score, agent.num_param_update) # record score every model update

            if agent.target_update_soft:
                if cnt % agent.target_update_freq_soft == 0:
                    agent.soft_target_network_update()
            else:
                if agent.num_param_update%agent.target_update_freq_hard==0:
                    agent.hard_target_network_update()

            cnt += 1

            if done:
                agent.score_list[e] = score
                if(e>98):
                    average_reward = np.sum(agent.score_list[e-99:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, score, agent.epsilon, average_reward))
                agent.writer.add_scalar("Average Reward", average_reward, e)
                agent.writer.add_scalar("Episodic Reward", score, e)
            
    #agent.save("cartpole-dqn.h5")
    #np.savetxt("dqn_scores.txt",agent.score_list,fmt='%.8f')

    # plt.figure(1)
    # #plt.subplot(121)
    # plt.plot(agent.score_list, label="episodic score")
    # #plt.subplot(122)
    # #plt.plot(average_reward, label="average score from all episodes")
    # #plt.legend()
    # plt.show()

    env.close()