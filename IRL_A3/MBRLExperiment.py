#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import time
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updatess = [1,3,5] 
    
    # IMPLEMENT YOUR EXPERIMENT HERE
    runtime_dict = {
        'Q-learning': [],
        'Dyna':[],
        'PS' : []
    }

    for wind_prop in wind_proportions:
        # Dyna curves (including Q-learning baseline at n_plan=0)
        lc_dyna = LearningCurvePlot(title=f"Dyna (wind={wind_prop})")
        x_axis = np.arange(0, n_timesteps, eval_interval)
        for n_plan in [0] + n_planning_updatess:
            avg_curve, runtimes = run_reps('dyna',
                                           wind_prop,
                                           n_plan,
                                           n_timesteps,
                                           eval_interval,
                                           n_repetitions,
                                           gamma,
                                           learning_rate,
                                           epsilon)
            label = f"n_plan={n_plan}"
            sm = smooth(avg_curve, window=5)
            lc_dyna.add_curve(x_axis, sm, label=label)

            if n_plan == 0:
                runtime_dict['Q-learning'].extend(runtimes)
            else:
                runtime_dict['Dyna'].extend(runtimes)
        lc_dyna.save(name=f'dyna_wind{wind_prop}.png')

        # Prioritized Sweeping curves
        lc_ps = LearningCurvePlot(title=f"Prioritized Sweeping (wind={wind_prop})")
        for n_plan in n_planning_updatess:
            avg_curve, runtimes = run_reps('ps',
                                           wind_prop,
                                           n_plan,
                                           n_timesteps,
                                           eval_interval,
                                           n_repetitions,
                                           gamma,
                                           learning_rate,
                                           epsilon)
            label = f"n_plan={n_plan}"
            sm = smooth(avg_curve, window=11) if len(avg_curve) >= 11 else avg_curve
            lc_ps.add_curve(x_axis, sm, label=label)

            runtime_dict['PS'].extend(runtimes)
        lc_ps.save(name=f'ps_wind{wind_prop}.png')

    # Comparison plots (best n_plan=5)
    best_n_plan = 5
    for wind_prop in wind_proportions:
        comp = LearningCurvePlot(title=f"Comparison (wind={wind_prop})")
        x_axis = np.arange(0, n_timesteps, eval_interval)

        # Q-learning baseline
        q_curve, _ = run_reps('dyna',
                              wind_prop,
                              0,
                              n_timesteps,
                              eval_interval,
                              n_repetitions,
                              gamma,
                              learning_rate,
                              epsilon)
        comp.add_curve(x_axis, smooth(q_curve, 5), label='Q-learning')

        # best Dyna
        dyna_curve, _ = run_reps('dyna',
                                 wind_prop,
                                 best_n_plan,
                                 n_timesteps,
                                 eval_interval,
                                 n_repetitions,
                                 gamma,
                                 learning_rate,
                                 epsilon)
        comp.add_curve(x_axis, smooth(dyna_curve, 5), label=f'Dyna (n_plan={best_n_plan})')

        # best PS
        ps_curve, _ = run_reps('ps',
                               wind_prop,
                               best_n_plan,
                               n_timesteps,
                               eval_interval,
                               n_repetitions,
                               gamma,
                               learning_rate,
                               epsilon)
        comp.add_curve(x_axis, smooth(ps_curve, 5), label=f'PS (n_plan={best_n_plan})')

        comp.save(name=f'comparison_wind{wind_prop}.png')

    # Runtime table
    print("\nAverage runtime per repetition (s):")
    for key, lst in runtime_dict.items():
        if lst:
            print(f"{key:<12}: {np.mean(lst):.2f} ± {np.std(lst):.2f}")


def run_reps(agent_type='dyna',
             wind_proportion=0.9,
             n_planning_updates=3,
             n_timesteps=10001,
             eval_interval=250,
             n_repetitions=20,
             gamma=1.0,
             learning_rate=0.2,
             epsilon=0.1):

    n_eval_points = int(np.floor((n_timesteps - 1) / eval_interval)) + 1
    avg_returns = np.zeros(n_eval_points)
    runtimes = []

    for repetition in range(n_repetitions):
        print(f"[{agent_type.upper()}] wind={wind_proportion:.1f}, n_plan={n_planning_updates} — Repetition {repetition + 1}/{n_repetitions}")

        env = WindyGridworld(wind_proportion=wind_proportion)
        if agent_type == 'dyna':
            agent = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
        elif agent_type == 'ps':
            agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)
        else:
            raise KeyError("Unknown agent type")

        t0 = time.time()

        # initial evaluation
        idx = 0
        avg_returns[idx] += agent.evaluate(env)
        idx += 1

        s = env.reset()
        for t in range(1, n_timesteps):
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            s = env.reset() if done else s_next

            if t % eval_interval == 0:
                test_env = WindyGridworld(wind_proportion)
                avg_returns[idx] += agent.evaluate(test_env)
                idx += 1

        runtimes.append(__import__('time').time() - t0)

    avg_returns /= n_repetitions
    return avg_returns, runtimes




if __name__ == '__main__':
    experiment()
