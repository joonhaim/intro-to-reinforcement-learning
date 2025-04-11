# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
###############################################################
# ShortCutExperiment.py
#
# Usage: python ShortCutExperiment.py
#    or: python ShortCutExperiment.py --qlearning --sarsa
#
# This script will run the experiments:
#   1. Q-Learning
#   2. SARSA
#   3. Windy environment (Q-Learning vs. SARSA)
#   4. Expected SARSA
#   5. n-step SARSA

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ShortCutAgents import (
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
    nStepSARSAAgent
)
from ShortCutEnvironment import (
    ShortcutEnvironment,
    WindyShortcutEnvironment
)

def run_experiment(
    agent_class,
    env_class,
    n_episodes,
    n_reps=1,
    agent_params=None,
    env_params=None
):
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}

    all_returns = np.zeros((n_reps, n_episodes))
    for rep in range(n_reps):
        env = env_class(**env_params)
        agent = agent_class(**agent_params)
        returns = agent.train(env, n_episodes)
        all_returns[rep, :] = returns
        print(f"[{agent_class.__name__}] Finished repetition {rep+1}/{n_reps}")

    avg_returns = np.mean(all_returns, axis=0)
    return avg_returns

def single_long_run(agent_class, env_class, n_episodes, agent_params=None, env_params=None):
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}

    env = env_class(**env_params)
    agent = agent_class(**agent_params)
    print(f"[{agent_class.__name__}] Starting single long run of {n_episodes} episodes...")
    returns = agent.train(env, n_episodes)
    print(f"[{agent_class.__name__}] Done.")
    return agent, env, returns

def plot_curves(curves, labels, title, filename, xlabel="Episode", ylabel="Average Return"):
    plt.figure()
    for data, label in zip(curves, labels):
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved figure: {filename}")

def experiment_qlearning():
    print("=== Q-Learning: Single long run (10,000 episodes) ===")
    q_agent_long, q_env_long, _ = single_long_run(
        QLearningAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for Q-Learning (ShortcutEnvironment):")
    q_env_long.render_greedy(q_agent_long.Q)

    print("=== Q-Learning: 100 reps of 1,000 episodes ===")
    n_reps = 100
    n_episodes = 1000
    q_curve = run_experiment(
        QLearningAgent,
        ShortcutEnvironment,
        n_episodes=n_episodes,
        n_reps=n_reps,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    q_curves_alphas = []
    labels_alphas = []
    for alpha in alpha_values:
        avg_curve = run_experiment(
            QLearningAgent,
            ShortcutEnvironment,
            n_episodes=n_episodes,
            n_reps=n_reps,
            agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": alpha, "gamma": 1.0}
        )
        q_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    plot_curves([q_curve], ["Q-Learning (alpha=0.1)"], "Q-Learning: 100 reps, 1000 episodes", "qlearning_100reps.png")
    plot_curves(q_curves_alphas, labels_alphas, "Q-Learning: alpha variation", "qlearning_alpha_variation.png")

def experiment_sarsa():
    print("=== SARSA: Single long run (10,000 episodes) ===")
    sarsa_agent_long, sarsa_env_long, _ = single_long_run(
        SARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for SARSA (ShortcutEnvironment):")
    sarsa_env_long.render_greedy(sarsa_agent_long.Q)

    print("=== SARSA: 100 reps of 1,000 episodes ===")
    n_reps = 100
    n_episodes = 1000
    sarsa_curve = run_experiment(
        SARSAAgent,
        ShortcutEnvironment,
        n_episodes=n_episodes,
        n_reps=n_reps,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    sarsa_curves_alphas = []
    labels_alphas = []
    for alpha in alpha_values:
        avg_curve = run_experiment(
            SARSAAgent,
            ShortcutEnvironment,
            n_episodes=n_episodes,
            n_reps=n_reps,
            agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": alpha, "gamma": 1.0}
        )
        sarsa_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    plot_curves([sarsa_curve], ["SARSA (alpha=0.1)"], "SARSA: 100 reps, 1000 episodes", "sarsa_100reps.png")
    plot_curves(sarsa_curves_alphas, labels_alphas, "SARSA: alpha variation", "sarsa_alpha_variation.png")

def experiment_windy():
    print("=== WindyShortcutEnvironment: Q-Learning vs. SARSA single run ===")

    windy_q_agent, windy_q_env, _ = single_long_run(
        QLearningAgent,
        WindyShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for Q-Learning (Windy):")
    windy_q_env.render_greedy(windy_q_agent.Q)

    windy_sarsa_agent, windy_sarsa_env, _ = single_long_run(
        SARSAAgent,
        WindyShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for SARSA (Windy):")
    windy_sarsa_env.render_greedy(windy_sarsa_agent.Q)

def experiment_expectedsarsa():
    print("=== Expected SARSA: Single long run (10,000 episodes) ===")
    esarsa_agent_long, esarsa_env_long, _ = single_long_run(
        ExpectedSARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for Expected SARSA (ShortcutEnvironment):")
    esarsa_env_long.render_greedy(esarsa_agent_long.Q)

    print("=== Expected SARSA: 100 reps of 1,000 episodes ===")
    n_reps = 100
    n_episodes = 1000
    esarsa_curve = run_experiment(
        ExpectedSARSAAgent,
        ShortcutEnvironment,
        n_episodes=n_episodes,
        n_reps=n_reps,
        agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    esarsa_curves_alphas = []
    labels_alphas = []
    for alpha in alpha_values:
        avg_curve = run_experiment(
            ExpectedSARSAAgent,
            ShortcutEnvironment,
            n_episodes=n_episodes,
            n_reps=n_reps,
            agent_params={"n_actions": 4, "n_states": 144, "epsilon": 0.1, "alpha": alpha, "gamma": 1.0}
        )
        esarsa_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    plot_curves([esarsa_curve], ["ExpectedSARSA (alpha=0.1)"], "Expected SARSA: 100 reps, 1000 episodes", "expected_sarsa_100reps.png")
    plot_curves(esarsa_curves_alphas, labels_alphas, "Expected SARSA: alpha variation", "expected_sarsa_alpha_variation.png")

def experiment_nstepsarsa():
    print("=== n-step SARSA: Single long run (10,000 episodes) ===")
    nsarsa_agent_long, nsarsa_env_long, _ = single_long_run(
        nStepSARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={"n_actions": 4, "n_states": 144, "n": 5, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
    )
    print("\nGreedy policy for n-step SARSA (n=5):")
    nsarsa_env_long.render_greedy(nsarsa_agent_long.Q)

    print("=== n-step SARSA: 100 reps of 1,000 episodes for various n ===")
    n_values = [1, 2, 5, 10, 25]
    nsarsa_curves = []
    nsarsa_labels = []
    n_reps = 100
    n_episodes = 1000

    for n_val in n_values:
        avg_curve = run_experiment(
            nStepSARSAAgent,
            ShortcutEnvironment,
            n_episodes=n_episodes,
            n_reps=n_reps,
            agent_params={"n_actions": 4, "n_states": 144, "n": n_val, "epsilon": 0.1, "alpha": 0.1, "gamma": 1.0}
        )
        nsarsa_curves.append(avg_curve)
        nsarsa_labels.append(f"n={n_val}")

    plot_curves(nsarsa_curves, nsarsa_labels, "n-step SARSA: 100 reps, 1000 episodes", "nstep_sarsa_n_variation.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qlearning", action="store_true", help="Run Q-Learning experiments")
    parser.add_argument("--sarsa", action="store_true", help="Run SARSA experiments")
    parser.add_argument("--windy", action="store_true", help="Run Windy (Q-Learning vs. SARSA) experiment")
    parser.add_argument("--expectedsarsa", action="store_true", help="Run Expected SARSA experiments")
    parser.add_argument("--nstepsarsa", action="store_true", help="Run n-step SARSA experiments")
    args = parser.parse_args()

    if not any(vars(args).values()):
        args.qlearning = True
        args.sarsa = True
        args.windy = True
        args.expectedsarsa = True
        args.nstepsarsa = True

    if args.qlearning:
        experiment_qlearning()
    if args.sarsa:
        experiment_sarsa()
    if args.windy:
        experiment_windy()
    if args.expectedsarsa:
        experiment_expectedsarsa()
    if args.nstepsarsa:
        experiment_nstepsarsa()

    print("\nAll experiments completed. Figures have been saved.")