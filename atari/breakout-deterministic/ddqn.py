# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import Logger
import cv2
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

EPISODES = 10000
WIDTH = 84
HEIGHT = 84
NUM_FRAMES = 4
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class CnnDDQNModel(nn.Module):
    def __init__(self, num_frames, action_size):
        super(CnnDDQNModel, self).__init__()
        self.num_frames = num_frames
        self.action_size = action_size

        self.conv1 = nn.Conv2d(in_channels=num_frames, out_channels=16, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):   
        # Forward pass
        x = self.relu(self.conv1(x))  # In: (80, 80, 4)  Out: (20, 20, 16)
        x = self.relu(self.conv2(x))  # In: (20, 20, 16) Out: (10, 10, 32)
        x = self.flatten(x)           # In: (10, 10, 32) Out: (3200,)
        x = self.fc(x)                # In: (3200,) Out: (4,) 
        return x

class DDQNAgent:
    def __init__(self, num_frames, action_size):
        self.folder = "./graphs/DDQN"
        self.cnn = True
        self.writer = SummaryWriter(log_dir=self.folder)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = [WIDTH, HEIGHT]
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.batch_size = 16
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.adam_learning_rate = 0.001
        self.rms_learning_rate = 0.00025
        self.rms_eps = 0.01
        self.rms_momentum = 0.9
        self.rms_alpha = 0.95
        self.tau = 0.1
        self.model = CnnDDQNModel(num_frames, self.action_size).to(self.device)
        self.target_model = CnnDDQNModel(num_frames, self.action_size).to(self.device)
        self.target_model.eval()
        self.score_list = np.zeros(EPISODES)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.adam_learning_rate)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.rms_learning_rate, eps=self.rms_eps, alpha=self.rms_alpha)
        #  momentum=self.rms_momentum, centered=True)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        self.target_update_soft = False
        self.target_update_freq_hard = 10000
        self.target_update_freq_soft = 10
        self.num_param_update = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def retrieve_samples(self):
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state, done = map(torch.cat, [*batch])
        return state, action, reward, next_state, done

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model.forward(state).squeeze()
            return act_values.max(0)[1].item()  # returns action

    def update(self):
        
        states, actions, rewards, next_states, dones = self.retrieve_samples()
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

        # Get indicies of actions with max value according to online model
        model_actions = self.model.forward(next_states).detach().max(1)[1]

        # gather Q values of states and above actions from target model
        target_q_values = self.target_model.forward(next_states).detach().gather(1, model_actions.unsqueeze(1)).squeeze()

        targets = rewards + self.gamma * target_q_values *(1-dones)

        # get q-values corresponding to actions at that step
        state_action_values = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss(state_action_values, targets)
        
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

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)     # convert to grayscale
        state = cv2.resize(state, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)   # Downsample to 105 x 80
        state = state[np.newaxis, np.newaxis, :, :]
        state = torch.FloatTensor(state).to(self.device)
        return state
    
    def show_image(self, img):
        plt.imshow(img, cmap="gray")
        plt.show()

if __name__ == "__main__":
    # env_id = gym.make('BreakoutDeterministic-v4')
    env_id = 'PongNoFrameskip-v4'
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    state_size = env.observation_space.shape     # 210 x 160 x 3
    action_size = env.action_space.n                # 4
    agent = DDQNAgent(NUM_FRAMES, action_size)
    # Logger.Logger(agent)    # Log agent parameters in a file
    average_reward = 0
    cnt = 0

    for e in range(EPISODES):
        state = env.reset()
        # state = agent.preprocess(state)
        # agent.show_image(state)
        score = 0
        done = False
        ep_cnt = 0
        
        while not done:
            #if(e>500):
            #    env.render()
            # accumulate NUM_FRAMES by performing some arbitrary action.
            # while state.size()[1] < NUM_FRAMES:
            #     action = 1
            #     next_state, reward, done, _ = env.step(action)
            #     # next_state = agent.preprocess(next_state)
            #     state = torch.cat([state, next_state], 1)
            # act according to the set of states, i.e. considers action to be made based on previous NUM_FRAMES frames
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            # reward = np.sign(reward)                        # reward clipping
            score += reward
            # new_frame = agent.preprocess(new_frame)
            # new_state = torch.cat([state, new_frame], 1)
            # next_state = new_state[:, 1:, :, :]
            # done = torch.FloatTensor([done]).to(agent.device)
            # reward = torch.FloatTensor([reward]).to(agent.device)
            # action = torch.LongTensor([action]).to(agent.device)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size:
                agent.update()
                agent.writer.add_scalar("Score per model update", score, agent.num_param_update) # record score every model update

            if agent.target_update_soft:
                if(cnt % agent.target_update_freq_soft == 0):
                    agent.soft_target_network_update()
            else:
                if agent.num_param_update%agent.target_update_freq_hard==0:
                    agent.hard_target_network_update()

            ep_cnt += 1
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