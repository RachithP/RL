# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import Logger
import cv2

EPISODES = 1000                 # add these kinda variable to a config file and import for better code structuring
NUM_FRAMES = 4

class CnnDQNModel(nn.Module):
    def __init__(self, input_shape, action_size, heads):
        super(CnnDQNModel, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.heads = heads

        self.input_layer = nn.Linear(self.state_size, 64)

        for i in range(self.heads):
            setattr(self, "hidden_%d" % i, nn.Linear(64, 64))
            setattr(self, "output_{}".format(i), nn.Linear(64, self.action_size))

    def forward(self, state):
        y = []
        w = F.relu(self.input_layer(state))
        for i in range(self.heads):
            x = getattr(self, "hidden_{}".format(i))(w)
            y.append(getattr(self, "output_%d" % i)(x))

        return y

class DQNAgent:
    def __init__(self, state_size, action_size, heads):
        self.folder = "./graphs/BstrpDDQN-1"
        self.writer = SummaryWriter(log_dir=self.folder)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.heads = heads
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.1
        self.model = CnnDQNModel(self.state_size, self.action_size, self.heads).to(self.device)
        self.target_model = CnnDQNModel(self.state_size, self.action_size, self.heads).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.smoothl1loss = nn.SmoothL1Loss()
        self.target_update_soft = False
        self.target_update_freq_hard = 100
        self.target_update_freq_soft = 10
        self.num_param_update = 0
    
    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.append((state, action, reward, next_state, done, mask))

    def act(self, state, rand_index):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model.forward(state)[rand_index].squeeze()
            return q_values.max(0)[1].item()  # returns action
    
    # def act(self, state, rand_index, cnt):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         Q_values = np.zeros(shape=(self.heads, self.action_size), dtype=np.float32)
    #         max_action = np.zeros(shape=(self.heads), dtype=np.int32)
    #         for i, qvalue in enumerate(self.model.forward(state)):  # we get self.heads number of outputs of actions for the same state
    #             Q_values[i, :] = qvalue.squeeze().cpu().numpy().reshape(1, self.action_size)
    #             max_action[i] = np.argmax(Q_values[i, :])
            
    #         # number of disagreements
    #         count = sum([1 for action in max_action if max_action[rand_index]!=action])
    #         agent.writer.add_scalar("#Disagreements per step", count, cnt)

    #         # get mode of actions chosen, most popular q-value
    #         mode, freq = stats.mode(max_action)
    #         agent.writer.add_scalar("Bool - Majority agree with current head?", mode==max_action[rand_index], cnt)

    #         return (self.model.forward(state)[rand_index].squeeze().cpu().numpy() + 5*Q_values.std(axis=0)).argmax()

    def estimate_uncertainty(self, states, rand_index):
        '''
            Estimating uncertainty in actions as it is a better measure as compared to measuring uncertainty in q-value estimations,
            because q values may be differently learnt and still have the relative difference in q-value(among different actions for the same state) the same?
        '''
        actions = np.zeros(shape=(self.batch_size,self.heads), dtype=np.float32)
        # q_values = np.zeros(shape=(self.batch_size,self.heads), dtype=np.float32)

        with torch.no_grad():
            for i, obs in enumerate(self.model.forward(states)):
                actions[:, i] = obs.max(1)[1].cpu().numpy().reshape(self.batch_size)
                # q_values[:, i] = obs.max(1)[0].cpu().numpy().reshape(self.batch_size)
            
            # q_values = np.around(q_values, decimals=1)
            a_mode, a_count = stats.mode(actions, axis=1) # return the mode value and its count
            # _, q_count = stats.mode(q_values, axis=1)
            # a_var = (1 - a_count/self.heads).mean()
            # q_var = (1 - q_count/self.heads).mean()

            # number of disagreements
            count = 0
            for row in actions:
                for action in row:
                    if(action!=row[rand_index]):
                        count+=1
            agent.writer.add_scalar("#Disagreements per step", count, self.num_param_update)
            
            # get mode of actions chosen, most popular q-value
            agent.writer.add_scalar("#Disagreements with mode per step",sum(a_mode.reshape(self.batch_size)!=actions[:, rand_index]),
            self.num_param_update)

        return count, sum(a_mode.reshape(self.batch_size)!=actions[:, rand_index])

    def update(self, rand_index):
        
        minibatch = random.sample(self.memory, self.batch_size)

        minibatch = np.reshape(minibatch, (len(minibatch), len(minibatch[0])))
        states = torch.FloatTensor(list(minibatch[:, 0])).reshape(len(minibatch), -1).to(self.device)
        actions = torch.LongTensor(list(minibatch[:, 1])).reshape(len(minibatch), -1).to(self.device)
        rewards = torch.FloatTensor(list(minibatch[:, 2])).reshape(
            len(minibatch), -1).to(self.device)
        next_states = torch.FloatTensor(list(minibatch[:, 3])).reshape(
            len(minibatch), -1).to(self.device)
        dones = torch.FloatTensor(list(minibatch[:, 4])).reshape(
            len(minibatch), -1).to(self.device)
        masks = torch.FloatTensor(list(minibatch[:, 5])).reshape(
            len(minibatch), -1).to(self.device)

        # train for all non-zero mask indices
        # mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
        # mask_tensor = torch.FloatTensor(mask.astype(np.int)).to(self.device)

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

            l2loss = self.loss(model_pred, targets)

            f_loss = masks[:, k]*l2loss

            losses.append(torch.sum(f_loss/masks[:, k].sum()))

        loss = sum(losses)/self.heads
        self.optimizer.zero_grad()
        loss.backward()

        # Clip grad norms on all parameter updates
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         # divide grads in core
        #         param.grad.data *=1.0/self.heads
        #         nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

        self.optimizer.step()
        self.num_param_update += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.estimate_uncertainty(states, rand_index)

    def soft_target_network_update(self):
        for q_weight, target_weight in zip(self.model.parameters(), self.target_model.parameters()):
            target_weight.data.copy_(self.tau * q_weight + (1-self.tau)*target_weight)

    def hard_target_network_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)     # convert to grayscale
        state = cv2.pyrDown(state, dstsize=(state.shape[1] //2, state.shape[0] //2))   # Downsample to 105 x 80
        state = torch.FloatTensor(state).to(self.device)
        return state
    
    def show_image(self, img):
        plt.imshow(img, cmap="gray")
        plt.show()

    
if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    state_size = env.observation_space.shape    # 210 x 160 x 3
    action_size = env.action_space.n                # 4
    agent = DQNAgent(state_size, action_size, heads=5)  
    Logger.Logger(agent)    # Log agent parameters in a file
    average_reward = 0
    cnt = 0

    for e in range(EPISODES):
        state = env.reset()     # type = np.ndarray
        state = agent.preprocess(state)
        rand_value_func_index = np.random.randint(low=0, high=agent.heads)      # choose a random head
        score = 0
        done = False
        e_a_var = 0
        e_q_var = 0
        ct = 0

        while not done:

            while state.size()[1] < NUM_FRAMES:
                action = 1
                next_state, reward, done, _ = env.step(action)
                next_state = agent.preprocess(next_state)
                state = torch.cat([state, next_state], 1)
            print(state.size())
            quit()
            #if(e>500):
            #    env.render()
            action = agent.act(state, rand_value_func_index)
            next_state, reward, done, _ = env.step(action)
            reward = np.sign(reward)
            mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
            agent.remember(state, action, reward, next_state, done, mask)
            state = next_state
            score += reward

            if len(agent.memory) > agent.batch_size:
                a_var, q_var = agent.update(rand_value_func_index)
                e_a_var += a_var
                e_q_var += q_var
                agent.writer.add_scalar("Score per model update", score, agent.num_param_update) # record score every model update

            if agent.target_update_soft:
                if cnt % agent.target_update_freq_soft == 0:
                    agent.soft_target_network_update()
            else:
                if agent.num_param_update%agent.target_update_freq_hard==0:
                    agent.hard_target_network_update()
            
            cnt += 1
            ct += 1

            if done:
                agent.score_list[e] = score
                if(e>98):
                    average_reward = np.sum(agent.score_list[e-99:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, score, agent.epsilon, average_reward))
                agent.writer.add_scalar("Average Reward", average_reward, e)
                agent.writer.add_scalar("Episodic Reward", score, e)
                agent.writer.add_scalar("#Disagreements per episode", e_a_var, e)
                agent.writer.add_scalar("#Disagreements in mode per Episode", e_q_var, e)
            
    #agent.save("cartpole-dqn.h5")
    #np.savetxt("dqn_scores.txt",agent.score_list,fmt='%.8f')
    # plt.figure(1)
    # #plt.subplot(121)
    # plt.plot(agent.score_list, label="episodic score")
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # #plt.subplot(122)
    # #plt.plot(average_reward, label="average score from all episodes")
    # #plt.legend()
    # plt.show()

    env.close()