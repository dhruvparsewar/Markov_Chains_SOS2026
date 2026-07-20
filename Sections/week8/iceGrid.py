import numpy as np

class StochasticGridworld:
    def __init__(self, slip_probability=0.2):
        """
        Initializes the 5x5 Stochastic Gridworld.
        Start: (0, 0)
        Goal: (4, 4) [Reward +10]
        Pits: (1, 1), (2, 1), (3, 1) [Reward -10]
        Step Penalty: -0.1
        """
        self.grid_size = 5
        self.slip_prob = slip_probability
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        self.pits = [(1, 1), (2, 1), (3, 1)]
        self.current_state = self.start_state
        
        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.actions = [0, 1, 2, 3]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        Takes an action and returns (next_state, reward, is_done).
        Incorporates the slip probability for stochastic transitions.
        """
        if self.current_state == self.goal_state or self.current_state in self.pits:
            return self.current_state, 0, True

        # Determine actual movement based on slip probability
        actual_action = action
        roll = np.random.rand()
        
        if roll < self.slip_prob:
            # Slipped! Move perpendicular
            if roll < self.slip_prob / 2:
                actual_action = (action - 1) % 4 # Slip left
            else:
                actual_action = (action + 1) % 4 # Slip right

        # Calculate new position
        x, y = self.current_state
        if actual_action == 0: y += 1   # Up
        elif actual_action == 1: x += 1 # Right
        elif actual_action == 2: y -= 1 # Down
        elif actual_action == 3: x -= 1 # Left

        # Keep within bounds
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))
        next_state = (x, y)

        # Calculate reward
        reward = -0.1 # Default step penalty
        is_done = False
        
        if next_state == self.goal_state:
            reward = 10.0
            is_done = True
        elif next_state in self.pits:
            reward = -10.0
            is_done = True

        self.current_state = next_state
        return next_state, reward, is_done

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table: Q[x, y, action] = 0.0
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions) # Explore
        else:
            x, y = state
            # Exploit: return the action with the max Q-value
            # Use random tie-breaking if multiple actions have the same max Q-value
            max_val = np.max(self.q_table[x, y])
            optimal_actions = np.where(self.q_table[x, y] == max_val)[0]
            return np.random.choice(optimal_actions) 

    def update_q_value(self, state, action, reward, next_state):
        """The Bellman Update."""
        x, y = state
        nx, ny = next_state
        
        best_next_q = np.max(self.q_table[nx, ny])
        
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a(Q(S', a)) - Q(S, A)]
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.alpha * td_error

    def train(self, episodes=5000):
        """Trains the agent over a specified number of episodes."""
        rewards_history = []
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            is_done = False
            
            while not is_done:
                action = self.choose_action(state)
                next_state, reward, is_done = self.env.step(action)
                
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
        return rewards_history

    def get_optimal_policy(self):
        """Extracts the optimal policy from the trained Q-table."""
        policy = np.zeros((self.env.grid_size, self.env.grid_size), dtype=int)
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                policy[x, y] = np.argmax(self.q_table[x, y])
        return policy

def visualize_policy(env, policy):
    """
    Prints a visual representation of the policy grid.
    Arrows represent the best action in each cell:
    ↑ (Up), → (Right), ↓ (Down), ← (Left).
    Marks Start (S), Goal (G), and Pits (X).
    """
    # Mapping actions to symbols: 0: Up, 1: Right, 2: Down, 3: Left
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    # We want to print top to bottom, so we iterate y in reverse order
    for y in range(env.grid_size - 1, -1, -1):
        row_str = ""
        for x in range(env.grid_size):
            state = (x, y)
            
            if state == env.start_state:
                # Mark start state, but show the policy direction next to it
                symbol = f"S{action_symbols[policy[x, y]]}" 
            elif state == env.goal_state:
                symbol = " G "
            elif state in env.pits:
                symbol = " X "
            else:
                symbol = f" {action_symbols[policy[x, y]]} "
                
            # Formatting to align columns
            row_str += f"[{symbol:^3}] "
        print(row_str)

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    print("--- Training with Slip Probability = 0.0 (Deterministic) ---")
    env_det = StochasticGridworld(slip_probability=0.0)
    agent_det = QLearningAgent(env_det, epsilon=1.0)
    agent_det.train(episodes=2000)
    policy_det = agent_det.get_optimal_policy()
    
    print("Optimal Policy (Deterministic):")
    visualize_policy(env_det, policy_det)
    
    print("\n--- Training with Slip Probability = 0.3 (Stochastic) ---")
    env_stoch = StochasticGridworld(slip_probability=0.3)
    agent_stoch = QLearningAgent(env_stoch, epsilon=1.0)
    agent_stoch.train(episodes=5000) # Takes a bit longer to converge with high noise
    policy_stoch = agent_stoch.get_optimal_policy()
    
    print("Optimal Policy (Stochastic - High Slip):")
    visualize_policy(env_stoch, policy_stoch)
    
    print("\nTraining complete. Q-tables populated.")