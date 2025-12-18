import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from BatchEnv import BatchEnv
import matplotlib.pyplot as plt

class DQN(nn.Module): # Neural net: state → 10 Q-values (one per action)
    def __init__(self, state_size=7, action_size=10): # 7 obs → 10 actions
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128) # Layer 1: 7→128 neuron
        self.fc2 = nn.Linear(128, 128) # Layer 2: 128→128
        self.fc3 = nn.Linear(128, action_size) # Output: 10 Q-values

    def forward(self, x): # Forward pass
        x = torch.relu(self.fc1(x)) # ReLU activation
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # Raw Q-values (no-softmax)

class DQNAgent:
    def __init__(self, state_size=7, action_size=10):
        self.state_size = state_size # 7 (your obs: C_A,B,D,U,V,P_B,P_D)
        self.action_size = action_size # 10 (0=stop, 1-9=feed pairs)
        self.memory = deque(maxlen=10000) # Experience replay buffer
        self.gamma = 0.999 # High discount: value future profits heavily
        self.epsilon = 1.0 # Exploration: 100% random at start
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.model = DQN(state_size, action_size) # Main network (trained)
        self.target_model = DQN(state_size, action_size) # Target network (updated every 100 steps - stabilizes training)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.steps = 0
        self.target_update = 100 # Update every 100 steps

    def log_q_values(self, state):
        with torch.no_grad():
            qvals = self.model(torch.FloatTensor(state).unsqueeze(0))
        print(f"Q[stop=0]: {qvals[0, 0]:.2f}, Q[max_feed]: {qvals.max():.2f}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, episode_idx=None, step_in_ep=None):
        C_A, C_B, C_D, C_U, V, P_B, P_D = state
        purity = C_D / (C_D + C_U + 1e-6)

        # Heuristic STOP exploration
        if episode_idx is not None and step_in_ep is not None:
            if step_in_ep >= 6 and purity > 0.6:
                if np.random.rand() < 0.5:
                    return 0  # STOP

        # ε-greedy fallback
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)

        with torch.no_grad():
            qvals = self.model(torch.FloatTensor(state).unsqueeze(0))
        return int(qvals.argmax().item())

        #if np.random.rand() <= self.epsilon: # ε-greedy return
            #return random.randrange(self.action_size) # Explore: random (0-9)
        #    if np.random.rand() < 0.2:
        #        return 0  # STOP
        #    return random.randrange(1, self.action_size)  # any feed action

        #state = torch.FloatTensor(state).unsqueeze(0) # Exploit: pick best Q-value action
        #with torch.no_grad():
        #    act_values = self.model(state)
        #act_values = self.model(state) # Get Q(s,a) for all 10 actions
        #return np.argmax(act_values.cpu().data.numpy()) # Pick highest Q-value

    def replay(self, batch_size=32): # Sample 32 experiences,
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size) # Random batch
        # Convert to tensors (states, actions, rewards, next_states, dones)
        # Extracts 5-tuple from each experience → stacks into tensors
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])) # [32, 7] 32, 7D states
        actions = torch.LongTensor(np.array([e[1] for e in minibatch])) # [32] (0-9)
        rewards = torch.FloatTensor(np.array([e[2] for e in minibatch])) # [32] profit/loss
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])) # [32, 7] next states
        dones = torch.BoolTensor(np.array([e[4] for e in minibatch])) # [32] flags
        # Use TARGET network for next_q (stable!)
        # Q-Learning loss: Q(s,a) ← r + γ max Q(s',a')
        current_q = self.model(states).gather(1, actions.unsqueeze(1)) # Q(s,a)
        next_q = self.target_model(next_states).max(1)[0].detach() # max Q(s', a')
        target_q = rewards + (self.gamma * next_q * ~dones) # Bellman
        loss = nn.MSELoss()(current_q.squeeze(), target_q) # Minimise loss
        self.optimizer.zero_grad() # Gradient descent
        loss.backward() # Compute gradients: ∂loss/∂weights
        self.optimizer.step() # Update weights: weights -= lr * gradients

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # 1.0 → 0.995 → 0.99 → ... → 0.01

        self.steps += 1
        if self.steps % self.target_update == 0: # ← UPDATE TARGET
            self.target_model.load_state_dict(self.model.state_dict())

        # Step 1: act() → random action (ε=1.0)
        # Step 2: remember() → store experience
        # Step 3: replay() →
            #├─ Sample 32 experiences
            #├─ Compute loss = [Q(s,a) - (r + γ max Q_target(s',a'))]²
            #├─ loss.backward() + optimizer.step() ← WEIGHTS UPDATE
            #├─ ε *= 0.995 ← LESS RANDOM
            # └─ Copy target net every 100 steps ← STABILITY
#Training loop
if __name__ == "__main__": # Test it works first
    env = BatchEnv(seed=42)
    state, info = env.reset()
    print("BatchEnv works:", state)

expert_actions = [4, 4, 1, 2, 1, 1, 1, 1, 0,
                  4, 4, 2, 5, 1, 1, 4, 1, 1, 0]

agent = DQNAgent()
episodes = 500
epsilons = []
scores = []
stop_frequencies = []
true_profits = []

for e in range(episodes): # 1000 episodes (batches)
    #use_expert = (np.random.rand() < 0.1)  # 10% expert episodes
    step_count = 0
    max_steps = 30
    state, info = env.reset() # New batch: 20L A, random prices
    trajectory = [] #
    total_reward = 0
    stop_count = 0
    episode_true_profit = 0.0
    #if use_expert:
    #    print(f"*** EXPERT Ep{e} actions: ", end="")

    while step_count < max_steps: # Max 30 steps per batch (safety)
        #action = agent.act(state) # ε-greedy action (0=stop, 1-9=feed)
        #if use_expert and step_count < len(expert_actions):
        #    action = expert_actions[step_count]
        #else:
        action = agent.act(state, episode_idx=e, step_in_ep=step_count)

        next_state, reward, done, truncated, info = env.step(action)
        #print(f"Action {action}: done={done}, purity={info.get('purity', 0):.2f}, V={next_state[4]:.1f}L")
        agent.remember(state, action, reward, next_state, done) # Store (s,a,r,s',done) in replay
        agent.replay() # Train on random batch

        #action = agent.act(state, episode_idx=e, step_in_ep=step_count)
        total_reward += reward # Sum batch profit
        if action == 0:
            stop_count += 1

        state = next_state
        step_count += 1

        if done or truncated:
            episode_true_profit = info.get('true_profit', 0.0)
            true_profits.append(episode_true_profit)  # ← CRITICAL LINE
            scores.append(total_reward)

            #print(f"APPENDED Ep{e}: true_profit={episode_true_profit:.1f}, list_len={len(true_profits)}")

            break  # Episode ends when action=0 (batch complete)
    #true_profits.append(episode_true_profit)
    trajectory.append(state.copy()) # ← STORE STATE
    #scores.append(total_reward) # Record episode profit
    stop_frequencies.append(stop_count / step_count)
    epsilons.append(agent.epsilon)

    if done:
        print(f"Ep{e}: true_profit={episode_true_profit:.1f}, shaped={total_reward:.1f}")

    #if done:
    #    print(f"Ep{e}: true_profit={true_profit:.1f}, shaped={total_reward:.1f}")

    if e % 100 == 0:
        print(f"Episode {e}, Avg Score: {np.mean(scores[-100:]):.2f}")
        print(f"Ep{e}, Final V: {state[4]:.1f}L, Reward: {total_reward:.1f}")
        volumes = [s[4] for s in trajectory]
        print(f"Ep{e} trajectory V: {volumes[:10]}... (len={len(trajectory)})")
        agent.log_q_values(state) # Diagnose: Q[stop] >> Q[feed]?
        print(f"Ep{e}, Final V: {state[4]:.1f}L, D: {state[2]:.2f}, ε: {agent.epsilon:.3f}")

# -------------------------------
# RANDOM POLICY BASELINE
# -------------------------------
#def run_random_policy(env, episodes=1000, max_steps=30):
#    random_scores = []

#    for ep in range(episodes):
#        state, _ = env.reset()
#       total_reward = 0

#        for t in range(max_steps):
#            action = np.random.randint(0, env.action_space.n)  # uniform random
#            next_state, reward, done, truncated, _ = env.step(action)
#            total_reward += reward
#            state = next_state

#            if done or truncated:
#                break

#        random_scores.append(total_reward)

#    return random_scores


# Run baseline
#random_env = BatchEnv(seed=123)
#random_scores = run_random_policy(random_env, episodes=episodes)

#print(f"Random policy avg reward: {np.mean(random_scores):.2f}")
print(f"DQN policy avg reward: {np.mean(scores[-100:]):.2f}")


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(scores)
plt.title('Episode Rewards (Noisy, Sparse Reward Effect)')
plt.ylabel('Profit/Episode')
plt.xlabel('Episode')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(stop_frequencies)
plt.title("STOP Action Frequency per Episode")
plt.xlabel("Episode")
plt.ylabel("STOP Frequency")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
# Left axis: reward
plt.plot(scores, label="Episode Reward", alpha=0.6)
plt.xlabel("Episode")
plt.ylabel("Reward")
# Right axis: epsilon
ax2 = plt.gca().twinx()
ax2.plot(epsilons, color="red", label="Epsilon (ε)", linestyle="--")
ax2.set_ylabel("Exploration Rate (ε)")
plt.title("ε-Decay vs Reward (Exploration–Exploitation Tradeoff)")
plt.grid(True)
# Legend handling for twin axes
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))

# Plot 1: True profit (thick blue line)
plt.subplot(1, 3, 1)
plt.plot(true_profits, 'b-', linewidth=2, label='True Profit', alpha=0.9)
plt.plot(scores, 'r--', linewidth=1, label='Shaped Reward', alpha=0.6)
plt.ylabel('$/Episode')
plt.legend()
plt.title('True vs Shaped')

# Plot 2: True profit only (clean)
plt.subplot(1, 3, 2)
plt.plot(true_profits, 'g-', linewidth=2)
plt.ylabel('True Profit ($)')
plt.title('True Profit Only')
plt.grid(True, alpha=0.3)

# Plot 3: Moving average
if len(true_profits) >= 100:
    ma100 = np.convolve(true_profits, np.ones(100)/100, mode='valid')
    plt.subplot(1, 3, 3)
    plt.plot(ma100, 'orange', linewidth=3)
    plt.title('100-Ep Moving Avg')
    plt.ylabel('True Profit ($)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#plt.figure(figsize=(12,6))
#plt.plot(scores, label="DQN (learning)", alpha=0.7)
#plt.plot(random_scores, label="Random Policy", linestyle="--", alpha=0.8)
#plt.axhline(0, color="black", linewidth=1)
#plt.xlabel("Episode")
#plt.ylabel("Episode Reward")
#plt.title("DQN vs Random Policy on BatchEnv")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()
