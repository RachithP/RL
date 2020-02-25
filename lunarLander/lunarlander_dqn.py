# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt

EPISODES = 512


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 64
        self.score_list = np.zeros(EPISODES)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.squeeze([ind[0] for ind in minibatch])
        actions = np.squeeze([ind[1] for ind in minibatch])
        rewards = np.squeeze([ind[2] for ind in minibatch])
        next_states = np.squeeze([ind[3] for ind in minibatch])
        dones = np.squeeze([ind[4] for ind in minibatch])

        state_action_values = self.model.predict_on_batch(states)

        targets = rewards + self.gamma * \
            np.amax(self.model.predict_on_batch(next_states), axis=1)*(1-dones)

        indices = np.array([i for i in range(self.batch_size)])

        state_action_values[[indices], [actions]] = targets

        self.model.fit(states, state_action_values, epochs=1, verbose=0)

        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         target = (reward + self.gamma *
        #                   np.amax(self.model.predict(next_state)[0]))
        #     target_f = self.model.predict(state)
        #     target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    average_reward = 0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0

        for time in range(1000):
            #if(e>500):
            #    env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if done:
                agent.score_list[e] = score
                if(e>99):
                    average_reward = np.sum(agent.score_list[e-100:]) / 100.0 # Average over last 100 episodes
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, score, agent.epsilon, average_reward))
                break

    # agent.save("cartpole-dqn.h5")
    # np.savetxt("scores.txt",agent.score_list,fmt='%.8f')

    plt.figure(0)
    #plt.subplot(121)
    plt.plot(agent.score_list, label="episodic score")
    #plt.subplot(122)
    #plt.plot(average_reward, label="average score from all episodes")
    #plt.legend()
    plt.show()

    env.close()
