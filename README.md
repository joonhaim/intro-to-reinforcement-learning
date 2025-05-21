# Introduction to Reinforcement Learning

Welcome to the repository for the *Introduction to Reinforcement Learning* course at **Leiden University**.

This repository contains all programming assignments, each exploring a different aspect of reinforcement learning — from bandit problems and dynamic programming to model-free and model-based learning.

---

## Authors

- **Adrien Joon-Ha Im**  
- **Bence Válint**

---

## Assignments Overview

---

### **Assignment 1 – Exploration Strategies in Bandits**

**Topic:** Action-value estimation and exploration in multi-armed bandits  
**Methods:**
- ε-Greedy  
- Optimistic Initialization  
- Upper Confidence Bound (UCB)

**Includes:**
- Strategy comparison  
- Average reward plots

📄 [Assignment 1A Report](IRL_A1/Assignment_1A_Report[FINAL].pdf)  
📄 [Assignment 1B Report](IRL_A1/Assignment_1B_Report[FINAL].pdf)

---

### **Assignment 2 – Model-Free Reinforcement Learning**

**Topic:** Learning through interaction without a model of the environment  
**Algorithms:**
- Q-Learning  
- SARSA  
- Expected SARSA  
- n-step SARSA

**Environment:**
- Custom GridWorld + Windy variant

**Includes:**
- Learning curves  
- Greedy policies  
- Parameter sensitivity analysis

📄 [Assignment 2 Report](IRL_A2/report.pdf)

---

### **Assignment 3 – Model-Based Reinforcement Learning**

**Topic:** Planning with an explicit environment model  
**Algorithms:**
- Dyna-Q (tabular model plus simulated planning)  
- Prioritized Sweeping (focused planning via priority queue)

**Environment:**
- Windy Gridworld (stochastic wind proportion 0.9 & 1.0)

**Includes:**
- Learning curves for varying planning depths (n = 0, 1, 3, 5)  
- Pure Q-learning baseline (n = 0)  
- Smooth vs. unsmoothed performance comparison  
- Runtime benchmarking  
- Comparison plots: Dyna vs. Prioritized Sweeping vs. Q-learning

📄 [Assignment 3 Report](IRL_A3/report.pdf)  

---

## Repository Structure

Each assignment is located in its own folder:

- `IRL_A1/`, `IRL_A2/`, `IRL_A3/`

Each folder typically contains:
- Agent & environment implementations  
- Experiment scripts  
- Visualizations and plots  
- Final report (`report.pdf`)
