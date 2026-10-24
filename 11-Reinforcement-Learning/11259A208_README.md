# 11. Reinforcement Learning

[Back to experiments](../README.md)

This experiment uses Q-learning in a six-state environment. The agent learns action values from rewards to reach the goal at state 5. It uses NumPy and does not require a CSV dataset.

[Python source](q_learning.py) | [Supplied text output](output.txt)

## Code

```python
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
```

## Output

```text
Q-Table:

[[  0.    0.    0.    0.   80.    0. ]
 [  0.    0.    0.   64.    0.  100. ]
 [  0.    0.    0.   64.    0.    0. ]
 [  0.   80.   51.2   0.   80.    0. ]
 [ 64.    0.    0.   64.    0.  100. ]
 [  0.    0.    0.    0.    0.    0. ]]
```

## Run

From this experiment folder, run: `python q_learning.py`.

The output above is the supplied Q-table. Training uses random exploration, so results may vary between runs.

