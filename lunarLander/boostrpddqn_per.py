# -*- coding: utf-8 -*-
import random, sys
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.PriorityBuffer import *

EPISODES = 512                 # add these kinda variable to a config file and import for better code structuring

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, heads):
        super(DQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.heads = heads

        self.input_layer = nn.Linear(self.state_size, 128)

        for i in range(self.heads):
            setattr(self, "hidden_%d" % i, nn.Linear(128, 128))
            setattr(self, "output_{}".format(i), nn.Linear(128, self.action_size))

    def forward(self, state):
        y = []
        w = F.relu(self.input_layer(state))
        for i in range(self.heads):
            x = getattr(self, "hidden_{}".format(i))(w)
            y.append(getattr(self, "output_%d" % i)(x))

        return y

class DQNAgent:
    def __init__(self, state_size, action_size, heads):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.heads = heads
        self.max_memory_size = 10000
        self.memory = PriorityBuffer(max_size=self.max_memory_size)
        self.batch_size = 64
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.1
        self.model = DQNModel(self.state_size, self.action_size, self.heads).to(self.device)
        self.target_model = DQNModel(self.state_size, self.action_size, self.heads).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mseloss = nn.MSELoss()
        self.smoothl1loss = nn.SmoothL1Loss()
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
            mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
            self.memory.store((state, action, reward, next_state, done, mask))

            if done:
                state = env.reset()
                state = np.reshape(state, [1, self.state_size])
            else:
                state = next_state
    
    
    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.store((state, action, reward, next_state, done, mask))

    def act(self, state, rand_index):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model.forward(state)[rand_index].squeeze()
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
        masks = torch.FloatTensor(list(minibatch_experiences[:, 5])).reshape(
            self.batch_size, -1).to(self.device).squeeze()
        weights = torch.FloatTensor(minibatch_IS_weights).reshape(
            self.batch_size, -1).to(self.device).squeeze()

        
        # get q-values corresponding to actions at that step
        state_action_values = self.model.forward(states)
        model_q_values = self.model.forward(next_states)
        target_q_values = self.target_model.forward(next_states)
        losses = []

        for k in range(self.heads):
            
            # get best action from K'th head according to online model's prediction
            model_actions = model_q_values[k].detach().max(1)[1]
            
            # gather Q values of states and above actions from K'th target model
            target_q_value = target_q_values[k].detach().gather(1, model_actions.unsqueeze(1)).squeeze()
            
            targets = rewards.squeeze() + self.gamma * target_q_value *(1-dones).squeeze()

            model_pred = state_action_values[k].gather(1, actions).squeeze()

            td_errors = targets - model_pred

            self.memory.update_priorities_batch(minibatch_tree_indices, np.abs(td_errors.cpu().detach().numpy()))

            l2loss = (weights*(td_errors)**2).mean()
            # l1loss = self.smoothl1loss(model_pred, targets)

            f_loss = masks[:, k]*l2loss

            losses.append(torch.sum(f_loss/masks[:, k].sum()))

        loss = sum(losses)/self.heads
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         # divide grads in core
        #         param.grad.data *=1.0/self.heads
        #         nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

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
    agent = DQNAgent(state_size, action_size, heads=5)
    # agent.load("./save/cartpole-dqn.h5")
    agent.run_agent_randomly(env)
    done = False
    average_reward = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        rand_value_func_index = np.random.randint(low=0, high=agent.heads)
        score = 0
        for time in range(1000):
            #if(e>500):
            #    env.render()
            action = agent.act(state, rand_value_func_index)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
            agent.remember(state, action, reward, next_state, done, mask)
            state = next_state
            score+= reward

            agent.update()

            # if(agent.num_param_update%agent.max_num_param_update==0):
            #     agent.hard_target_network_update()

            if(time % 4 == 0):
                agent.soft_target_network_update()

            if done:
                agent.score_list[e] = score
                if(e>98):
                    average_reward = np.sum(agent.score_list[e-99:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, score, agent.epsilon, average_reward))
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