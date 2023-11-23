import numpy as np
import tensorflow as tf
from collections import deque
import random
import gym

'''
CartPole-v1 dari Gym, 
agen harus menjaga tongkat (pole) agar tidak jatuh.
'''
# env
env = gym.make('CartPole-v1')

memory = deque(maxlen=2000)
gamma = 0.95  
epsilon = 1.0 
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64


'''

924724
679568
000232
323400

'''

# Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_dim=env.observation_space.shape[0], activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state))

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])

    # limit 500 steps
    for step in range(500):  
        action = choose_action(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        
        memory.append((state, action, reward, next_state, done))

        state = next_state

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_, d in minibatch:
                target = r
                if not d:
                    target = r + gamma * np.amax(model.predict(s_)[0])
                target_full = model.predict(s)
                target_full[0][a] = target
                model.fit(s, target_full, epochs=1, verbose=0)

        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
