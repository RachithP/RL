# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

EPISODES = 1000


class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.1
        self.online_model = self._build_model()
        self.target_model = self._build_model()
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.online_model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def update(self):

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.squeeze([ind[0] for ind in minibatch])
        actions = np.squeeze([ind[1] for ind in minibatch])
        rewards = np.squeeze([ind[2] for ind in minibatch])
        next_states = np.squeeze([ind[3] for ind in minibatch])
        dones = np.squeeze([ind[4] for ind in minibatch])

        online_model_actions = np.argmax(self.online_model.predict_on_batch(next_states), axis=1)

        ind = np.array([i for i in range(self.batch_size)])

        target_q_values = self.target_model.predict_on_batch(next_states)[ind, online_model_actions]

        targets = rewards + self.gamma * target_q_values * (1-dones)

        state_action_values = self.online_model.predict_on_batch(states)

        state_action_values[[ind], [actions]] = targets

        self.online_model.fit(states, state_action_values, epochs=1, verbose=0)

        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         target_action =  np.argmax(self.online_model.predict(next_state)[0])
        #         target_q_value = self.target_model.predict(next_state)[0][target_action]
        #         target = reward + self.gamma * np.amax(target_q_value)

        #     target_f = self.online_model.predict(state)
        #     target_f[0][action] = target
        #     self.online_model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        q_network_theta = self.online_model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta, target_model_theta):
            target_weight = target_weight * (1-self.tau) + q_weight * self.tau
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)

    def load(self, name):
        self.online_model.load_weights(name)

    def save(self, name):
        self.online_model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DoubleDQNAgent(state_size, action_size)
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

            if(time%4==0):
                agent.update_target_network()

            if len(agent.memory) > agent.batch_size:
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
    #plt.subplot(122)
    #plt.plot(average_reward, label="average score from all episodes")
    #plt.legend()
    plt.show()

    env.close()
