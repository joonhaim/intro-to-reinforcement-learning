# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
#
# Usage: python ShortCutExperiment.py
#    or: python ShortCutExperiment.py --qlearning --sarsa
#
# This script will run the experiments:
#   1. Q-Learning (--qlearning)
#   2. SARSA (--sarsa)
#   3. Windy environment (Q-Learning vs. SARSA) (--windy)
#   4. Expected SARSA (--expectedsarsa)
#   5. n-step SARSA (--nstepsarsa)


import argparse
import numpy as np
import matplotlib.pyplot as plt

from ShortCutAgents import (QLearningAgent, SARSAAgent,ExpectedSARSAAgent,nStepSARSAAgent)
from ShortCutEnvironment import (ShortcutEnvironment,WindyShortcutEnvironment)

def smooth(data, window=10):
    """
    Applies smoothing to the array.
    """
    if window < 2:
        return data
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumulative_sum[window:] - cumulative_sum[:-window]) / float(window)
    pad_length = len(data) - len(smoothed)
    return np.concatenate((np.full(pad_length, smoothed[0]), smoothed))


def run_experiment(
    agent_class,
    env_class,
    n_episodes,
    n_reps=1,
    agent_params=None,
    env_params=None,
    smoothing_window=1
):
    """
    Runs a given agent on a given environment over multiple repetitions.
    Returns the averaged return per episode (optionally smoothed).

    :param agent_class: A reference to one of the agent classes (QLearningAgent, etc.)
    :param env_class: A reference to the environment class (ShortcutEnvironment, etc.)
    :param n_episodes: Number of episodes per repetition.
    :param n_reps: Number of independent repetitions.
    :param agent_params: Dictionary of agent hyperparameters.
    :param env_params: Dictionary of environment construction parameters.
    :param smoothing_window: For simple moving average smoothing of the final curve.
    :return: 1D numpy array of length n_episodes (averaged & smoothed).
    """
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}

    all_returns = np.zeros((n_reps, n_episodes))
    for rep in range(n_reps):
        # Fresh environment & agent each repetition
        env = env_class(**env_params)
        agent = agent_class(**agent_params)
        returns = agent.train(env, n_episodes)
        all_returns[rep, :] = returns
        print(f"[{agent_class.__name__}] Finished repetition {rep+1}/{n_reps}")

    avg_returns = np.mean(all_returns, axis=0)
    # Optionally smooth
    avg_returns_smoothed = smooth(avg_returns, window=smoothing_window)
    return avg_returns_smoothed


def single_long_run(agent_class, env_class, n_episodes, agent_params=None, env_params=None):
    """
    Runs a single 'long' experiment (e.g., 10,000 episodes) and returns
    the trained agent & environment. Useful for visualizing the greedy policy.
    """
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
    """
    Utility to plot multiple learning curves on one figure.
    curves: list of 1D arrays
    labels: list of strings
    """
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


############################
# Separate experiment fns
############################

def experiment_qlearning():
    """
    Runs Q-Learning experiments:
      (a) Single long run (10,000 episodes) + show greedy policy
      (b) 100 reps of 1,000 episodes
      (c) alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== Q-Learning: Single long run (10,000 episodes) ===")
    q_agent_long, q_env_long, q_returns_long = single_long_run(
        QLearningAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 12 * 12,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
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
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        },
        smoothing_window=10
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
            agent_params={
                "n_actions": 4,
                "n_states": 144,
                "epsilon": 0.1,
                "alpha": alpha,
                "gamma": 1.0
            },
            smoothing_window=10
        )
        q_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    # Plot
    plot_curves(
        [q_curve],
        ["Q-Learning (alpha=0.1)"],
        "Q-Learning: 100 reps, 1000 episodes",
        "qlearning_100reps.png"
    )
    plot_curves(
        q_curves_alphas,
        labels_alphas,
        "Q-Learning: alpha variation",
        "qlearning_alpha_variation.png"
    )


def experiment_sarsa():
    """
    Runs SARSA experiments:
      (a) Single long run (10,000 episodes) + show greedy policy
      (b) 100 reps of 1,000 episodes
      (c) alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== SARSA: Single long run (10,000 episodes) ===")
    sarsa_agent_long, sarsa_env_long, sarsa_returns_long = single_long_run(
        SARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
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
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        },
        smoothing_window=10
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
            agent_params={
                "n_actions": 4,
                "n_states": 144,
                "epsilon": 0.1,
                "alpha": alpha,
                "gamma": 1.0
            },
            smoothing_window=10
        )
        sarsa_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    # Plot
    plot_curves(
        [sarsa_curve],
        ["SARSA (alpha=0.1)"],
        "SARSA: 100 reps, 1000 episodes",
        "sarsa_100reps.png"
    )
    plot_curves(
        sarsa_curves_alphas,
        labels_alphas,
        "SARSA: alpha variation",
        "sarsa_alpha_variation.png"
    )


def experiment_windy():
    """
    WindyShortcutEnvironment single runs: Q-Learning vs. SARSA
    """
    print("=== WindyShortcutEnvironment: Q-Learning vs. SARSA single run ===")

    # Q-Learning single run
    windy_q_agent, windy_q_env, _ = single_long_run(
        QLearningAgent,
        WindyShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
    )
    print("\nGreedy policy for Q-Learning (Windy):")
    windy_q_env.render_greedy(windy_q_agent.Q)

    # SARSA single run
    windy_sarsa_agent, windy_sarsa_env, _ = single_long_run(
        SARSAAgent,
        WindyShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
    )
    print("\nGreedy policy for SARSA (Windy):")
    windy_sarsa_env.render_greedy(windy_sarsa_agent.Q)


def experiment_expectedsarsa():
    """
    Expected SARSA experiments on ShortcutEnvironment:
      (a) Single long run (10,000 episodes) + show policy
      (b) 100 reps of 1,000 episodes
      (c) alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== Expected SARSA: Single long run (10,000 episodes) ===")
    esarsa_agent_long, esarsa_env_long, esarsa_returns_long = single_long_run(
        ExpectedSARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
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
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        },
        smoothing_window=10
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
            agent_params={
                "n_actions": 4,
                "n_states": 144,
                "epsilon": 0.1,
                "alpha": alpha,
                "gamma": 1.0
            },
            smoothing_window=10
        )
        esarsa_curves_alphas.append(avg_curve)
        labels_alphas.append(f"alpha={alpha}")

    # Plot
    plot_curves(
        [esarsa_curve],
        ["ExpectedSARSA (alpha=0.1)"],
        "Expected SARSA: 100 reps, 1000 episodes",
        "expected_sarsa_100reps.png"
    )
    plot_curves(
        esarsa_curves_alphas,
        labels_alphas,
        "Expected SARSA: alpha variation",
        "expected_sarsa_alpha_variation.png"
    )


def experiment_nstepsarsa():
    """
    n-step SARSA experiment on ShortcutEnvironment:
      (a) Single long run (10,000 episodes, default n=5)
      (b) 100 reps of 1,000 episodes for n in [1, 2, 5, 10, 25]
    """
    print("=== n-step SARSA: Single long run (10,000 episodes) ===")
    nsarsa_agent_long, nsarsa_env_long, nsarsa_returns_long = single_long_run(
        nStepSARSAAgent,
        ShortcutEnvironment,
        n_episodes=10000,
        agent_params={
            "n_actions": 4,
            "n_states": 144,
            "n": 5,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0
        }
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
            agent_params={
                "n_actions": 4,
                "n_states": 144,
                "n": n_val,
                "epsilon": 0.1,
                "alpha": 0.1,
                "gamma": 1.0
            },
            smoothing_window=10
        )
        nsarsa_curves.append(avg_curve)
        nsarsa_labels.append(f"n={n_val}")

    # Plot
    plot_curves(
        nsarsa_curves,
        nsarsa_labels,
        "n-step SARSA: 100 reps, 1000 episodes",
        "nstep_sarsa_n_variation.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qlearning", action="store_true", help="Run Q-Learning experiments")
    parser.add_argument("--sarsa", action="store_true", help="Run SARSA experiments")
    parser.add_argument("--windy", action="store_true", help="Run Windy (Q-Learning vs. SARSA) experiment")
    parser.add_argument("--expectedsarsa", action="store_true", help="Run Expected SARSA experiments")
    parser.add_argument("--nstepsarsa", action="store_true", help="Run n-step SARSA experiments")

    args = parser.parse_args()

    # If no flags are given, run everything:
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

