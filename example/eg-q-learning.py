import numpy as np

# env
num_states = 5
num_actions = 4
rewards = np.random.randint(-5, 6, size=(num_states, num_actions))

# intial optimal Q = 10
Q_star = np.ones([num_states, num_actions]) * 10  

learning_rate = 0.8
discount_factor = 0.9
num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    state = np.random.randint(0, num_states)
    for step in range(max_steps_per_episode):
        action = np.argmax(Q_star[state, :])
        new_state = np.random.choice(range(num_states))
        reward = rewards[state, action]
        Q_star[state, action] = Q_star[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q_star[new_state, :]) - Q_star[state, action])
        state = new_state
        if state == 4:
            break

print("nilai q:")
print(Q_star)
