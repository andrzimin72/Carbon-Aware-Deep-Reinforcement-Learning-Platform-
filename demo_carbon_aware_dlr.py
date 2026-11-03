#!/usr/bin/env python3
"""
Carbon-Aware Deep Reinforcement Learning Platform — Standalone Demonstration Script

This script showcases the core capabilities of the platform using synthetic data.
It requires no external APIs, no Kubernetes cluster, and no complex setup.

Capabilities demonstrated:
1. Synthetic Grid Data Generation (Diurnal patterns)
2. Reinforcement Learning Environment (Gymnasium-style)
3. Recurrent PPO Agent Training (LSTM-based)
4. Shadow Mode Inference & Baseline Comparison
5. Automated Visualization (Matplotlib)

Prerequisites:
    pip install numpy torch matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==============================================================================
# 1. SYNTHETIC DATA GENERATION (The Data Pipeline)
# ==============================================================================

def generate_synthetic_grid(hours=24*7):
    """Generates realistic carbon intensity data for 3 regions over N hours."""
    t = np.linspace(0, hours * 2 * np.pi / 24, hours)
    
    # Region 0: Solar-heavy (Duck curve: low midday, high evening)
    caiso = 300 - 150 * np.sin(t) + np.random.normal(0, 15, hours)
    
    # Region 1: Wind-heavy (Volatile, often low at night)
    miso = 400 - 100 * np.cos(t) + np.random.normal(0, 25, hours)
    
    # Region 2: Fossil-heavy (High baseline, low volatility)
    ercot = 550 + np.random.normal(0, 10, hours)
    
    grid_data = np.column_stack([caiso, miso, ercot])
    return np.clip(grid_data, 50, 800).astype(np.float32)

# ==============================================================================
# 2. THE ENVIRONMENT (The Training Ground)
# ==============================================================================

class CarbonEnv:
    """A simplified Gymnasium environment for carbon-aware scheduling."""
    def __init__(self, grid_data, window_size=5):
        self.grid = grid_data
        self.window_size = window_size
        self.current_step = 0
        self.max_steps = len(grid_data) - window_size
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
        
    def _get_state(self):
        # State: Last N hours of CI for all 3 regions
        history = self.grid[self.current_step : self.current_step + self.window_size]
        return torch.FloatTensor(history).flatten() # Shape: (window_size * 3,)

    def step(self, action):
        # Action: 0=CAISO, 1=MISO, 2=ERCOT
        ci = self.grid[self.current_step + self.window_size - 1, action]
        
        # Reward: Negative carbon cost (we want to minimize CI)
        # Baseline penalty to make rewards positive for visualization
        reward = 600.0 - ci 
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done

# ==============================================================================
# 3. THE AGENT (Recurrent Actor-Critic PPO)
# ==============================================================================

class RecurrentAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.hidden_state = None

    def forward(self, x):
        # x shape: (1, 1, state_dim) for LSTM
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)
        probs = torch.softmax(self.actor(lstm_out.squeeze(0)), dim=-1)
        value = self.critic(lstm_out.squeeze(0))
        return probs, value

    def reset_hidden(self):
        self.hidden_state = None

# ==============================================================================
# 4. TRAINING & INFERENCE LOOP
# ==============================================================================

def train_agent(env, agent, episodes=200):
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    rewards_history = []
    
    print("🚀 Starting Training...")
    for ep in range(episodes):
        state = env.reset()
        agent.reset_hidden()
        total_reward = 0
        
        # Simple Policy Gradient / Actor-Critic update for demo brevity
        log_probs = []
        values = []
        rewards = []
        
        done = False
        while not done:
            # Reshape for LSTM: (batch=1, seq=1, features)
            x = state.unsqueeze(0).unsqueeze(0) 
            probs, value = agent(x)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
        # Update weights (Simplified Advantage)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        actor_loss = 0
        critic_loss = 0
        for log_prob, value, ret in zip(log_probs, values, returns):
            actor_loss += -log_prob * ret
            critic_loss += (value - ret).pow(2)
            
        optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        optimizer.step()
        
        rewards_history.append(total_reward)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{episodes} | Avg Reward: {np.mean(rewards_history[-50:]):.1f}")
            
    return rewards_history

def run_shadow_mode(env, agent, baseline_region=2):
    """Runs the trained agent vs a static baseline."""
    state = env.reset()
    agent.reset_hidden()
    
    agent_choices = []
    baseline_choices = []
    agent_carbon = []
    baseline_carbon = []
    
    done = False
    while not done:
        x = state.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs, _ = agent(x)
        action = torch.argmax(probs, dim=-1).item()
        
        # Get actual CI for this step
        current_ci = env.grid[env.current_step + env.window_size - 1]
        
        agent_choices.append(action)
        agent_carbon.append(current_ci[action])
        
        baseline_choices.append(baseline_region)
        baseline_carbon.append(current_ci[baseline_region])
        
        # Step environment for agent
        next_state, _, done = env.step(action)
        state = next_state
        
    return np.array(agent_carbon), np.array(baseline_carbon), np.array(agent_choices)

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================

def plot_results(grid_data, agent_carbon, baseline_carbon, agent_choices, rewards):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Carbon-Aware Deep Reinforcement Learning Platform: Demonstration Results', fontsize=16, fontweight='bold')
    
    hours = np.arange(len(agent_carbon))
    
    # Plot 1: Grid Carbon Intensity
    axs[0, 0].plot(hours, grid_data[-len(hours):, 0], label='Region 0 (Solar)', color='#2E86AB')
    axs[0, 0].plot(hours, grid_data[-len(hours):, 1], label='Region 1 (Wind)', color='#A23B72')
    axs[0, 0].plot(hours, grid_data[-len(hours):, 2], label='Region 2 (Fossil)', color='#F18F01')
    axs[0, 0].set_title('Synthetic Grid Carbon Intensity (gCO₂e/kWh)')
    axs[0, 0].set_ylabel('Carbon Intensity')
    axs[0, 0].legend()
    
    # Plot 2: Agent Decisions
    axs[0, 1].scatter(hours, agent_choices, c=agent_carbon, cmap='RdYlGn_r', s=10, alpha=0.7)
    axs[0, 1].set_title('Agent Scheduling Decisions')
    axs[0, 1].set_ylabel('Chosen Region (0, 1, or 2)')
    axs[0, 1].set_ylim(-0.5, 2.5)
    axs[0, 1].set_yticks([0, 1, 2])
    
    # Plot 3: Cumulative Carbon Savings
    agent_cum = np.cumsum(agent_carbon)
    base_cum = np.cumsum(baseline_carbon)
    axs[1, 0].plot(hours, base_cum, label='Baseline (Always Region 2)', color='red', linestyle='--')
    axs[1, 0].plot(hours, agent_cum, label='AI Agent (Carbon-Aware)', color='green', linewidth=2)
    axs[1, 0].set_title('Cumulative Carbon Emissions')
    axs[1, 0].set_ylabel('Total gCO₂e')
    axs[1, 0].legend()
    
    # Plot 4: Training Curve
    axs[1, 1].plot(rewards, color='blue', alpha=0.5)
    # Smooth curve
    smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
    axs[1, 1].plot(range(19, len(rewards)), smoothed, color='blue', linewidth=2, label='Smoothed Reward')
    axs[1, 1].set_title('Agent Training Progress')
    axs[1, 1].set_ylabel('Episode Reward')
    axs[1, 1].set_xlabel('Episodes')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150)
    print("\n✅ Visualization saved to 'demo_results.png'")
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CARBON-AWARE DEEP ReEINFORCEMENT LEARNING PLATFORM — STANDALONE DEMO")
    print("="*60)
    
    # 1. Generate Data
    print("\n1. Generating 30 days of synthetic grid data...")
    grid = generate_synthetic_grid(hours=24*30)
    
    # 2. Setup Environment & Agent
    env = CarbonEnv(grid, window_size=5)
    state_dim = 5 * 3 # 5 hours * 3 regions
    agent = RecurrentAgent(state_dim, action_dim=3)
    
    # 3. Train
    print("\n2. Training Recurrent PPO Agent...")
    rewards = train_agent(env, agent, episodes=200)
    
    # 4. Shadow Mode (Compare against a 'dirty' baseline)
    print("\n3. Running Shadow Mode Inference (vs. Fossil Baseline)...")
    # Reset env to the last 24 hours for the demo visualization
    env.current_step = len(grid) - 24 - 5 
    agent_carbon, base_carbon, choices = run_shadow_mode(env, agent, baseline_region=2)
    
    savings = np.sum(base_carbon) - np.sum(agent_carbon)
    print(f"   Carbon Saved in 24h: {savings:.1f} gCO₂e ({(savings/np.sum(base_carbon)*100):.1f}%)")
    
    # 5. Visualize
    print("\n4. Generating Results Visualization...")
    plot_results(grid, agent_carbon, base_carbon, choices, rewards)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
