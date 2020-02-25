# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt
from keras import backend as K
from scipy import stats
import h5py
import time

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.validation_fraction = 0.3
        self.batch_size = 32
        self.model = self._build_model()
        self.score_list = np.zeros(EPISODES)
        self.average_reward = np.zeros(EPISODES)
        self.episodic_variance = np.zeros([2, EPISODES])
        self.episodic_loss = np.zeros(EPISODES)
        self.epistemic_uncertainty = np.zeros(EPISODES)
        self.var = np.zeros((1, self.action_size))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        dropout = 0.1
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        lam = 2
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        var = self.get_uncertainty(state)
        #print(var[0])
        self.var = np.array(var[0])
        act_values = self.model.predict(state)
        #print('1: ', act_values[0])
        #print('2: ', act_values[0]+lam*var[0])
        return np.argmax(act_values[0]+lam*var[0])  # returns action
        
    def get_uncertainty(self, state):
        model = self.model
        T = 20
        predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

        Yhat = np.array([predict_stochastic([state, 1]) for _ in range(T)])

        Yhat = np.reshape(Yhat,(T,-1))
        #temp = np.sum(Yhat,axis=0)
        #E = 1/T*temp # average estimated output given T random samples from stochastic forward passes through DQN
        
        temp = np.around(Yhat,decimals=1)
        _, count = stats.mode(temp,axis=0)

        Var = 1 - count/T
        
        return Var

    def replay(self):

        batch = random.sample(self.memory, self.batch_size + int(len(self.memory)*self.validation_fraction))

        minibatch = batch[0:self.batch_size]
        val_data = batch[self.batch_size:]
        loss = 0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            train = self.model.fit(state, target_f, epochs=1, verbose=0)
            loss += train.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Performing uncertainty on val data
        var = 0
        for state, action, reward, next_state, done in val_data:
            var += np.mean(self.get_uncertainty(state))

        var /= len(val_data)

        return loss, var

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False

    for e in range(EPISODES):
        start = time.time()
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episodic_variance = np.zeros((1, agent.action_size))

        for t in range(500):    
            # if(e>500):
                # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episodic_variance += agent.var
            if done:
                #print(e)
                agent.score_list[e] = t
                if(e>99):
                    agent.average_reward[e] = np.mean(agent.score_list[e-100:])
                agent.episodic_variance[0][e] = episodic_variance[0][0]
                agent.episodic_variance[1][e] = episodic_variance[0][1]
                break
        if len(agent.memory) > agent.batch_size + int(len(agent.memory)*agent.validation_fraction):
            loss, var = agent.replay()
            agent.episodic_loss[e] = loss
            agent.epistemic_uncertainty[e] = var
            print("episode: {}/{}, score: {}, e: {:.2}, Average reward: {}, Episodic variance: {}, loss: {}, val uncertainty: {}"
                .format(e, EPISODES, t, agent.epsilon, agent.average_reward[e], episodic_variance, loss, var))

        if(e%100==199):
            plt.figure(0)
            plt.clf()
            plt.subplot(121)
            plt.plot(agent.score_list, label="episodic score")
            plt.subplot(122)
            plt.plot(agent.episodic_loss, label="episodic loss")

            plt.figure(1)
            plt.clf()
            plt.subplot(121)
            plt.plot(agent.episodic_variance[0,:], label="action[0] episodic uncertainty")
            plt.subplot(122)
            plt.plot(agent.episodic_variance[1,:],
                     label="action[1] episodic uncertainty")

            plt.figure(2)
            plt.clf()
            plt.plot(agent.epistemic_uncertainty, label="epistemic uncertainty")

            plt.draw()
            plt.pause(0.1)
        
    print('time:', time.time()-start, 'seconds')
    agent.save("results/uncertainty/un_cartpole-dqn-10000.h5")
    np.savetxt("results/uncertainty/un_scores_1000.txt",
               agent.score_list, fmt='%.8f')
    np.savetxt("results/uncertainty/un_variances_1000.txt",
               (agent.episodic_variance[0, :].T, agent.episodic_variance[1, :].T))
    np.savetxt("results/uncertainty/un_epistemic_uncertainty_1000.txt",
               agent.epistemic_uncertainty, fmt='%.8f')

    env.close()
