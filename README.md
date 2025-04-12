# Introduction to Reinforcement Learning – Course Assignments

This repository contains all programming assignments for the *Introduction to Reinforcement Learning* course at **Leiden University**.

Each assignment explores different aspects of reinforcement learning, ranging from bandit problems and dynamic programming to model-free learning.

### Authors

- Adrien Joon-Ha Im
- Bence Vállint
---

## 📚 Assignments

### ✅ **Assignment 1 – Exploration Strategies in Bandits**
Focuses on action-value estimation and exploration methods in multi-armed bandit problems:
- ε-Greedy
- Optimistic Initialization
- Upper Confidence Bound (UCB)

Includes plots of average rewards and comparison of strategies.

---

### ✅ **Assignment 2 – Model-Free Reinforcement Learning**
Implements and compares several model-free RL algorithms in a custom gridworld:
- Q-Learning
- SARSA
- Expected SARSA
- n-step SARSA
- Windy Environment variant

Includes learning curves, greedy policies, and parameter experiments.

---

### 📁 Structure

Each assignment is in its own folder (e.g., `IRL_A1`, `IRL_A2`) and typically includes:
- Agent/environment implementations
- Experiment scripts
- Plots and visualizations
- A final report (`report.pdf`) with results and interpretations

---

## ▶️ How to Run

Install dependencies:
```bash
pip install numpy matplotlib scipy
