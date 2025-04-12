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

**Topic:** Planning using a full model of the environment  
**Techniques:**
- Value Iteration  
- Policy Iteration

**Includes:**
- Convergence analysis  
- Optimal value functions  
- Policy visualizations

📄 [Assignment 3 Report](report.pdf)

---

## Repository Structure

Each assignment is located in its own folder:

- `IRL_A1/`, `IRL_A2/`, `IRL_A3/`

Each folder typically contains:
- Agent & environment implementations  
- Experiment scripts  
- Visualizations and plots  
- Final report (`report.pdf`)
