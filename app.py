from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

learning_rate = 0.8
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000
max_steps_per_episode = 100

# labirin env
num_states = 25
num_actions = 4 # Atas, Kanan, Bawah, Kiri

# Reward: -1 langkah biasa, 10 untuk tujuan
rewards = np.full((5, 5), -1)
rewards[4, 4] = 10

# nyimpen q
Q = np.zeros([num_states, num_actions])

# e-greedy
def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(range(num_actions)) 
    else:
        return np.argmax(Q[state, :]) 

def perform_q_learning():
    for episode in range(num_episodes):
        state = 0  
        for step in range(max_steps_per_episode):
            action = choose_action(state, Q, epsilon)

            if action == 0:  # Atas
                new_state = state - 5 if state >= 5 else state
            elif action == 1:  # Kanan
                new_state = state + 1 if (state + 1) % 5 != 0 else state
            elif action == 2:  # Bawah
                new_state = state + 5 if state < 20 else state
            else:  # Kiri
                new_state = state - 1 if state % 5 != 0 else state

            reward = rewards[new_state // 5, new_state % 5]

            Q[state, action] = Q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state  
            if state == 24:
                break

@app.route('/start', methods=['POST'])
def start_q_learning():
    perform_q_learning()
    return jsonify({'message': 'Q-learning started'})

@app.route('/get_best', methods=['GET'])
def get_best_policy():
    best_policy = np.argmax(Q, axis=1).tolist()
    return jsonify({'best_policy': best_policy})

if __name__ == '__main__':
    app.run(debug=True)
