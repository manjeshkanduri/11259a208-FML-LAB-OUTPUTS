import numpy as np

R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

Q = np.zeros_like(R, dtype=float)

gamma = 0.8
episodes = 1000

for _ in range(episodes):
    state = np.random.randint(0, 6)

    while state != 5:
        actions = np.where(R[state] >= 0)[0]
        action = np.random.choice(actions)
        next_state = action

        Q[state, action] = (
            R[state, action] + gamma * np.max(Q[next_state])
        )

        state = next_state

print("Q-Table:\n")
print(Q)
