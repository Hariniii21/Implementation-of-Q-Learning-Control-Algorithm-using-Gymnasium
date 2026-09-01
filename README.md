# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To develop a reinforcement learning agent using the Q-Learning control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function through trial and error and determine the best actions to reach the goal state while avoiding holes and maximizing the total reward.

## Software Requirements

* Python 3.x
* Google Colab / Jupyter Notebook
* Gymnasium
* NumPy
* Matplotlib

## Environment Description
A custom 4×4 FrozenLake environment is used.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm



## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
# Write your code here
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # Select action using epsilon-greedy policy
        action = choose_action(state, epsilon)

        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Find maximum Q-value for the next state
        best_next_q = np.max(Q[next_state])

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * best_next_q - Q[state, action]
        )

        # Move to next state
        state = next_state
        total_reward += reward

        # End episode if terminated or truncated
        if terminated or truncated:
            break

    # Store episode reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Calculate state values and learned policy
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

```
---

## Output

<img width="930" height="627" alt="image" src="https://github.com/user-attachments/assets/b1598ff5-432a-458d-b20e-ac2bbe2430db" />

<img width="937" height="581" alt="image" src="https://github.com/user-attachments/assets/bb8d8ed0-85f6-4c2a-8d35-27e82d51be02" />

## Result

Thus, the Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned Q-values for different state action pairs and obtained a policy for reaching the goal while avoiding the holes.

## Inference

The experiment shows that Q-Learning gradually improves the agent's decision-making through repeated interaction with the environment. Initially, the agent explores different actions, but as training progresses, it learns better Q-values and selects more suitable actions. The learned policy helps the agent reach the goal state while reducing the chances of falling into holes. The learning curve also shows the improvement in the agent's performance over time.


---

