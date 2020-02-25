# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

EPISODES = 1000

class BDQNAgent:
    def __init__(self, state_size, action_size, heads):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.heads = heads
        self.models = []
        for _ in range(self.heads):
            self.models.append(self._build_model())
        self.score_list = np.zeros(EPISODES)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.append((state, action, reward, next_state, done, mask))

    ''' Not choosing actions this way
    def act(self, state, model):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = model.predict(state)
        return np.argmax(act_values[0])  # returns action
    '''

    def update(self):

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done, mask in minibatch:
            ind = np.squeeze(np.where(mask == 1))
            target = reward
            if(ind.shape!=()):
                for i in ind:
                    if not done:
                        target = (reward + self.gamma *
                                np.amax(self.models[i].predict(next_state)[0]))
                    target_f = self.models[i].predict(state)
                    target_f[0][action] = target
                    self.models[i].fit(state, target_f, epochs=1, verbose=0)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def load(self, name, model):
        model.load_weights(name)

    def save(self, name, model):
        model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = BDQNAgent(state_size, action_size, heads=5)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    average_reward = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        value_func_index = np.random.randint(low=0, high=agent.heads)
        for time in range(500):
            #if(e>500):
            #    env.render()
            # action = agent.act(state, agent.models[value_func_index])
            action = np.argmax(agent.models[value_func_index].predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            # mask = np.random.randint(low=0, high=2, size=agent.heads)
            mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
            agent.remember(state, action, reward, next_state, done, mask)
            state = next_state

            if len(agent.memory) > agent.batch_size*agent.batch_size:
                agent.update()

            if done:
                agent.score_list[e] = time
                if(e > 99):
                    average_reward = np.sum(agent.score_list[e-100:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, time, agent.epsilon, average_reward))
                break

    #agent.save("cartpole-dqn.h5")
    #np.savetxt("dqn_scores.txt",agent.score_list,fmt='%.8f')

    plt.figure(1)
    #plt.subplot(121)
    plt.plot(agent.score_list, label="episodic score")
    #plt.legend()
    #plt.subplot(122)
    #plt.plot(average_reward, label="average score from all episodes")
    #plt.legend()
    plt.show()

    env.close()