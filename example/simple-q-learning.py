import numpy as np

# env
num_states = 5
num_actions = 4

# contoh reward -5 dan 5
rewards = np.random.randint(-5, 6, size=(num_states, num_actions))

# q-learning
Q_q_learning = np.zeros([num_states, num_actions])

learning_rate = 0.8
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = np.random.randint(0, num_states)
    for step in range(max_steps_per_episode):
        action = np.argmax(Q_q_learning[state, :] + np.random.randn(1, num_actions) * epsilon)
        new_state = np.random.choice(range(num_states))
        reward = rewards[state, action]
        Q_q_learning[state, action] = Q_q_learning[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q_q_learning[new_state, :]) - Q_q_learning[state, action])
        '''
         diguankan untuk uppdate nilai Q dari keadaan dan aksi tertentu. ini menggabungkan informasi reward yang diperoleh dari aksi yang diambil (reward), nilai Q max dari keadaan baru (np.max(Q_q_learning[new_state, :])), serta nilai Q saat ini untuk aksi yang diambil (Q_q_learning[state, action])..
        '''
        state = new_state
        if state == 4:
            break

print("nilai q:")
print(Q_q_learning)
