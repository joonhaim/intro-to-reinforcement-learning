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
import matplotlib.patches as patches

from ShortCutAgents import (QLearningAgent, SARSAAgent,ExpectedSARSAAgent,nStepSARSAAgent)
from ShortCutEnvironment import (ShortcutEnvironment,WindyShortcutEnvironment)

#-------------------------
# Helper Functions
#-------------------------

def smooth(data, window=10):
    """Smooth curve using a moving average"""
    if window < 2:
        return data
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumulative_sum[window:] - cumulative_sum[:-window]) / float(window)
    padding = np.full(len(data) - len(smoothed), smoothed[0])
    return np.concatenate((padding, smoothed))


def run_experiment(agent_class, env_class, n_episodes, n_reps=1, agent_params=None, env_params=None, smoothing_window=1):
    """
    Runs a given agent on a given environment over multiple repetitions.
    Returns the averaged return per episode (optionally smoothed).

    agent_class: A reference to one of the agent classes (QLearningAgent, etc.)
    env_class: A reference to the environment class (ShortcutEnvironment, etc.)
    n_episodes: Number of episodes per repetition.
    n_reps: Number of independent repetitions.
    agent_params: Dictionary of agent hyperparameters.
    env_params: Dictionary of environment construction parameters.
    smoothing_window: For simple moving average smoothing of the final curve.
    returns 1D numpy array of length n_episodes (averaged & smoothed).
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
    Runs a single 'long' experiment (10,000 episodes) and returns the trained agent & environment.
    """
    if agent_params is None:
        agent_params = {}
    if env_params is None:
        env_params = {}

    env = env_class(**env_params)
    agent = agent_class(**agent_params)

    print(f"[{agent_class.__name__}] Starting single long run of {n_episodes} episodes...")
    agent.train(env, n_episodes)
    print(f"[{agent_class.__name__}] Done.")
    return agent, env


def plot_curves(curves, labels, title, filename, xlabel="Episode", ylabel="Cumulative reward"):
    """Plot and save learning curves"""
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


def save_greedy_trajectory(Q, env, filename, title="Greedy Trajectory"):
    """
    Renders & saves a plot of the environment
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    arrow_map = ["↑", "↓", "←", "→"]
    # shape (r, c)
    greedy_actions = np.argmax(Q, axis=1).reshape((env.r, env.c))
    # we will compute the path as a list of (x, y) positions

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.c)
    ax.set_ylim(0, env.r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    # 1) Draw each cell
    for y in range(env.r):
        for x in range(env.c):
            cell_type = env.s[y, x]
            color = 'white'
            if cell_type == 'G':
                color = 'green'
            elif cell_type == 'C':
                color = 'red'
            rect = patches.Rectangle((x, env.r - 1 - y), 1, 1,
                                     linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.3)
            ax.add_patch(rect)

    # 2) Draw the best-action arrows
    for y in range(env.r):
        for x in range(env.c):
            a = greedy_actions[y, x]
            # Optionally skip if Q-values are all zero
            # if not any(Q[y*env.c + x]):  # or check np.max
            #     continue
            arrow_char = arrow_map[a]
            ax.text(x + 0.5, env.r - 1 - y + 0.5, arrow_char,
                    ha='center', va='center', fontsize=12)

    # 3) Simulate from the start to get the actual path
    env.reset()  # Needed to initialize env state

    # Force bottom start
    env.x = env.c // 6
    env.y = 5 * env.r // 6 - 1
    env.starty = env.y

    path_coords = []
    max_steps = 200  # to prevent infinite loop if we never reach the goal
    for _ in range(max_steps):
        sx, sy = env.x, env.y
        path_coords.append((sx + 0.5, env.r - 1 - sy + 0.5))
        if env.done():
            break
        s_idx = env.state()
        a = np.argmax(Q[s_idx])
        env.step(a)
        if env.done():
            # Add final position
            path_coords.append((env.x + 0.5, env.r - 1 - env.y + 0.5))
            break

    # 4) Draw the path (thick line connecting each point in path_coords)
    if len(path_coords) > 1:
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, color='blue', linewidth=3, marker='o', markersize=4)

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved greedy trajectory as: {filename}")

#-------------------------
# Experiment Functions
#-------------------------

def experiment_qlearning():
    """
    Runs Q-Learning experiments:
      1. Single long run (10,000 episodes) + show greedy policy
      2. 100 reps of 1,000 episodes
      3. alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== Q-Learning: Single long run (10,000 episodes) ===")
    agent, env = single_long_run(
        QLearningAgent,
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
    print("\nGreedy policy for Q-Learning (ShortcutEnvironment):")
    env.render_greedy(agent.Q)

    save_greedy_trajectory(
        agent.Q,
        env,
        "trajectory_qlearning.png",
        title="Greedy Trajectory - Q-Learning"
    )

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
        print(f"=== Q-Learning: 100 reps of 1,000 episodes for alpha {alpha} ===")
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
      1. Single long run (10,000 episodes) + show greedy policy
      2. 100 reps of 1,000 episodes
      3. alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== SARSA: Single long run (10,000 episodes) ===")
    agent, env = single_long_run(
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
    env.render_greedy(agent.Q)

    save_greedy_trajectory(agent.Q, env, "trajectory_sarsa.png", title="Greedy Trajectory - SARSA")


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
    agent, env = single_long_run(
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
    env.render_greedy(agent.Q)

    save_greedy_trajectory(agent.Q, env, "trajectory_qlearning_windy.png", title="Greedy Trajectory - Q-Learning (Windy)")

    # SARSA single run
    agent, env = single_long_run(
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
    env.render_greedy(agent.Q)

    save_greedy_trajectory(agent.Q, env, "trajectory_sarsa_windy.png", title="Greedy Trajectory - SARSA (Windy)")


def experiment_expectedsarsa():
    """
    Expected SARSA experiments on ShortcutEnvironment:
      1. Single long run (10,000 episodes) + show policy
      2. 100 reps of 1,000 episodes
      3. alpha in [0.01, 0.1, 0.5, 0.9]
    """
    print("=== Expected SARSA: Single long run (10,000 episodes) ===")
    agent, env = single_long_run(
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
    env.render_greedy(agent.Q)

    save_greedy_trajectory(agent.Q, env, "trajectory_expectedsarsa.png", title="Greedy Trajectory - Expected SARSA")

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
      1. Single long run (10,000 episodes, default n=5)
      2. 100 reps of 1,000 episodes for n in [1, 2, 5, 10, 25]
    """
    print("=== n-step SARSA: Single long run (10,000 episodes) ===")
    agent, env = single_long_run(
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
    env.render_greedy(agent.Q)

    save_greedy_trajectory(agent.Q, env, "trajectory_nstepsarsa.png", title="Greedy Trajectory - n-step SARSA (n=5)")

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
    plot_curves(nsarsa_curves, nsarsa_labels,"n-step SARSA: 100 reps, 1000 episodes",
                "nstep_sarsa_n_variation.png")



#-------------------------
# Main function
#-------------------------
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