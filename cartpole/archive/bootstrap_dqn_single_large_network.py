# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Activation, Input
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

EPISODES = 1000

''' Not Using as of now
class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.
    
    # Arguments
       index: loss_weight index that needs to be updated
    """

    def __init__(self, index):
        self.index = index

    def on_epoch_begin(self, epoch, logs=None):

        # Decrease weight for binary cross-entropy loss
        sess = K.get_session()
        self.model.loss_weight[self.index].load(1.0, sess)

        print("loss_weight["+str(self.index)+"] = ", self.model.loss_weight[self.index].eval(sess))
'''


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
        self.model = self._build_model()
        self.score_list = np.zeros(EPISODES)

    def _build_model(self):
        # Define input layer
        input_layer = Input(shape=(self.state_size,))

        x = Dense(24, activation='relu')(input_layer)
        x = Dense(24, activation='relu')(x)
        output = [None] * self.heads
        for i in range(self.heads):
            head = Dense(24, activation='relu')(x)
            output[i] = Dense(self.action_size, activation='linear')(head)

        model = Model(inputs=[input_layer], outputs=[output[i] for i in range(self.heads)])

        # Initialize loss weights
        model.loss_weight = [None] * self.heads
        for i in range(self.heads):
            model.loss_weight[i] = tf.Variable(0, trainable=False, name='loss_weight_'+str(i), dtype=tf.float32)

        # Compile model
        model.compile(loss=['mse' for _ in range(self.heads)], optimizer=Adam(lr=self.learning_rate, clipnorm=5.), loss_weights=[model.loss_weight[i] for i in range(self.heads)])

        return model

    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.append((state, action, reward, next_state, done, mask))

    ''' Not using to decide actions
    def act(self, state, model):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = model.predict(state)
        return np.argmax(act_values[0])  # returns action
    '''

    def update(self):
        sess = K.get_session()
        for i in range(self.heads):
            # sample with replacement
            minibatch = random.sample(self.memory, self.batch_size)

            # set weight of loss of current head as 1
            self.model.loss_weight[i].load(1.0, sess)

            for state, action, reward, next_state, done, mask in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                            np.amax(self.model.predict(next_state)[i][0]))
                target_f = self.model.predict(state)
                target_f[i][0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

            # reset the weight of loss on the current head    
            self.model.loss_weight[i].load(0.0, sess)
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
        # sample an head from uniform distribution
        value_func_index = np.random.randint(low=0, high=agent.heads)
        
        for time in range(500):
            # action = agent.act(state, agent.models[value_func_index])
            action = np.argmax(agent.model.predict(state)[value_func_index][0])
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            # mask = np.random.randint(low=0, high=2, size=agent.heads)
            mask = np.random.binomial(n=1, p=0.5, size=agent.heads)
            agent.remember(state, action, reward, next_state, done, mask)
            state = next_state

            if len(agent.memory) > agent.batch_size*agent.heads:
                agent.update()

            if done:
                agent.score_list[e] = time
                if(e > 99):
                    average_reward = np.sum(agent.score_list[e-100:])/100.0
                print("episode: {}/{}, score: {}, e: {:.2}, average_reward: {}"
                      .format(e, EPISODES, time, agent.epsilon, average_reward))
                break

    #agent.save("cartpole-bdqn_single_large_network.h5")
    #np.savetxt("bdqn_scores.txt",agent.score_list,fmt='%.8f')

    plt.figure(1)
    #plt.subplot(121)
    plt.plot(agent.score_list, label="episodic score")
    #plt.legend()
    #plt.subplot(122)
    #plt.plot(average_reward, label="average reward last 100 episodes")
    #plt.legend()
    plt.show()

    env.close()
