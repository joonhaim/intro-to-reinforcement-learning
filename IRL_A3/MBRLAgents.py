#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements

        # Value table
        self.Q_sa = np.zeros((n_states, n_actions))

        # Tab model
        self.n_sas = np.zeros((n_states, n_actions, n_states), dtype=int) #counting transitions
        self.R_sum_sas = np.zeros((n_states, n_actions, n_states), dtype=float) #sums of rewards

        #Track visited s,a pairs for random sampling in planning phase
        self.visited_sa = [] #list of (s,a)
        self._visited_set = set() # set



    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        q = self.Q_sa[s]
        max_q = np.max(q)
        greedy = np.flatnonzero(q == max_q)
        return int(np.random.choice(greedy))
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add Dyna update

        # 1) Real Q-learning update
        target = r + self.gamma * (0.0 if done else np.max(self.Q_sa[s_next]))
        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])

        # 2) Update counts and rewards
        self.n_sas[s, a, s_next] += 1
        self.R_sum_sas[s, a, s_next] += r

        # 3) Keep track of visited pairs
        if (s,a) not in self._visited_set:
            self._visited_set.add((s,a))
            self.visited_sa.append((s,a))

        # 4) Plan update
        for _ in range(n_planning_updates):
            if not self.visited_sa:
                break
            s_sim, a_sim = self.visited_sa[np.random.randint(len(self.visited_sa))]
            counts = self.n_sas[s_sim, a_sim]
            total = counts.sum()
            if total == 0:
                continue
            probs = counts / total
            s_sim_next = int(np.random.choice(self.n_states, p=probs))
            r_sim = self.R_sum_sas[s_sim, a_sim, s_sim_next] / counts[s_sim_next]
            tgt = r_sim + self.gamma * np.max(self.Q_sa[s_sim_next])
            self.Q_sa[s_sim, a_sim] += self.learning_rate * (tgt - self.Q_sa[s_sim, a_sim])



    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = int(np.argmax(self.Q_sa[s]))
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        return float(np.mean(returns))

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements

        self.Q_sa = np.zeros((n_states, n_actions))

        self.n_sas = np.zeros((n_states, n_actions, n_states), dtype=int)
        self.R_sum_sas = np.zeros((n_states, n_actions, n_states), dtype=float)
        self.predecessors = [set() for _ in range(n_states)]
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        q = self.Q_sa[s]
        max_q = q.max()
        greedy_actions = np.flatnonzero(q == max_q)
        return int(np.random.choice(greedy_actions))

    def update(self,s,a,r,done,s_next,n_planning_updates):
        
        # TO DO: Add Prioritized Sweeping code
        
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue

        # 1) Standard Q-learning upd.
        old_q = self.Q_sa[s, a]
        target = r + self.gamma * (0.0 if done else np.max(self.Q_sa[s_next]))
        td_err = target - old_q
        self.Q_sa[s, a] += self.learning_rate * td_err

        #2) Update model
        self.n_sas[s, a, s_next] += 1
        self.R_sum_sas[s, a, s_next] += r
        self.predecessors[s_next].add((s,a))

        #3) push current (s,a) on priority queue
        if abs(td_err) > self.priority_cutoff:
            self.queue.put((-abs(td_err), (s, a)))

        #4 Plan updates
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (s_p, a_p) = self.queue.get()

            counts = self.n_sas[s_p, a_p]
            total = counts.sum()
            if total == 0:
                continue


            probs = counts / total
            s_p_next = int(np.random.choice(self.n_states, p=probs))
            r_p = self.R_sum_sas[s_p, a_p, s_p_next] / counts[s_p_next]

            old_q_p = self.Q_sa[s_p, a_p]
            tgt_p = r_p + self.gamma * np.max(self.Q_sa[s_p_next])
            td_err_p = tgt_p - old_q_p
            self.Q_sa[s_p, a_p] += self.learning_rate * (tgt_p - self.Q_sa[s_p, a_p])

            # backward priorities
            for (s_bar, a_bar) in self.predecessors[s_p]:
                cnt_bar = self.n_sas[s_bar, a_bar, s_p]
                if cnt_bar == 0:
                    continue
                r_bar = self.R_sum_sas[s_bar, a_bar, s_p] / cnt_bar
                td_err_bar = r_bar + self.gamma * np.max(self.Q_sa[s_p]) - self.Q_sa[s_bar, a_bar]
                if abs(td_err_bar) > self.priority_cutoff:
                    self.queue.put((-abs(td_err_bar), (s_bar, a_bar)))


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = int(np.argmax(self.Q_sa[s]))
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        return float(np.mean(returns))

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
