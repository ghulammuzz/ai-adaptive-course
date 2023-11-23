import numpy as np

# env (3x3 grid world) atas, kiri, kanan, bawah
num_states = 9
num_actions = 4  

# Reward 10 
rewards = np.array([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, 10]  
])

Q = np.zeros([num_states, num_actions])

learning_rate = 0.8
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000
max_steps_per_episode = 100

# e-greedy
def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(range(num_actions)) 
    else:
        # choose the best action
        return np.argmax(Q[state, :])

for episode in range(num_episodes):
    state = 0  
    for step in range(max_steps_per_episode):
        action = choose_action(state, Q, epsilon)

        new_state = action  
        reward = rewards[state // 3, state % 3] 

        # q reoair
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state 

        if state == 8:
            break

print("nilai Q abis pinter:")
print(Q)

# - leraning rate
# - discount factor
# - epsilon