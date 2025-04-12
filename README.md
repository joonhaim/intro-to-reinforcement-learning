# Introduction to Reinforcement Learning – Course Assignments

This repository contains all programming assignments for the *Introduction to Reinforcement Learning* course at **Leiden University**.

Each assignment explores different aspects of reinforcement learning, ranging from bandit problems and dynamic programming to model-free and model-based learning.

### Authors

- Adrien Joon-Ha Im  
- Bence Válint  

---

## 📚 Assignments

### **Assignment 1 – Exploration Strategies in Bandits**
Focuses on action-value estimation and exploration methods in multi-armed bandit problems:
- ε-Greedy  
- Optimistic Initialization  
- Upper Confidence Bound (UCB)  

Includes plots of average rewards and comparison of strategies.  
📄 [1A Report](IRL_A1/Assignment_1A_Report[FINAL].pdf)
📄 [1B Report](IRL_A1/Assignment_1B_Report[FINAL].pdf)

---

### **Assignment 2 – Model-Free Reinforcement Learning**
Implements and compares several model-free RL algorithms in a custom gridworld:
- Q-Learning  
- SARSA  
- Expected SARSA  
- n-step SARSA  
- Windy Environment variant  

Includes learning curves, greedy policies, and parameter experiments.  
📄 [Read the report](IRL_A2/report.pdf)

---

### **Assignment 3 – Model-Based Reinforcement Learning**
Applies value iteration and policy iteration to tabular gridworlds using a full model of the environment.  
Analyzes convergence, optimal value functions, and policies.  
📄 [Read the report](mbrl_assignment.pdf)

---

## 📁 Structure

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
