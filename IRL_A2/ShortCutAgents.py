import numpy as np
class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        #Initialize Q(s,a) = 0
        self.Q = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state):
        # TO DO: Implement policy
        # Epsilon Greedy Action Selection
        if np.random.rand() < self.epsilon:
            #Explore
            return np.random.randint(self.n_actions)
        else:
            #Exploit
            return np.argmax(self.Q[state,:])


    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state,:])

        td_error = target - self.Q[state,action]
        self.Q[state,action] += self.alpha * td_error

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode

        episode_returns = []
        for _ in range(n_episodes):
            env.reset()
            s = env.state()
            done = env.done()
            ep_return = 0

            while not done:
                a = self.select_action(s)
                r = env.step(a)
                sp = env.state()
                done = env.done()

                #Q-learning update
                self.update(s, a, r, sp, done)

                s=sp
                ep_return += r

            episode_returns.append(ep_return)

        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        #Initialization Q(s,a) = 0
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # TO DO: Implement policy

        #Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])

    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env, n_episodes):
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset()
            s = env.state()
            done = env.done()
            ep_return = 0
            a = self.select_action(s)

            while not done:
                r = env.step(a)
                sp = env.state()
                done = env.done()

                # Choose next action (on-policy)
                ap = self.select_action(sp)

                # SARSA update
                self.update(s, a, r, sp, ap, done)

                s = sp
                a = ap
                ep_return += r

            episode_returns.append(ep_return)
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        #initialize Q(s,a) = 0
        self.Q = np.zeros((n_states, n_actions))


    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])

    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        if done:
            target = reward
        else:
            #Compute expected value of Q(s',a') under epsilon greedy
            q_next = self.Q[next_state, :]
            best_a = np.argmax(q_next)

            #Prob of choosing best action under eps-greedy
            prob_best = 1.0 - self.epsilon + (self.epsilon / self.n_actions)

            #Prob of choosing any other action
            prob_other = (1 - prob_best) / self.n_actions

            # Compute expected value
            expected_next_reward = 0.0
            for a_prime in range(self.n_actions):
                reward = q_next[a_prime]
                if a_prime == best_a:
                    expected_next_reward += prob_best * reward
                else:
                    expected_next_reward += prob_other * reward
            target = reward + self.gamma * expected_next_reward
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset()
            s = env.state()
            ep_return = 0

            while not env.done():
                a = self.select_action(s)
                r = env.step(a)
                sp = env.state()

                #Update
                self.update(s, a, r, sp, env.done())

                s=sp
                ep_return += r
            episode_returns.append(ep_return)

            print("Completed episode: ", _)
        return episode_returns


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        #initialize Q(s,a) =0
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state,:])

    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        length = len(states)

        updates_to_perform = 0
        if length >= self.n:
            updates_to_perform = 1
        if done:
            updates_to_perform = length

        for _ in range(updates_to_perform):
            s0 = states[0]
            a0 = actions[0]

            G = 0.0
            discount = 1.0

            #sum of discounted rewards
            for reward in rewards[0:min(self.n,len(rewards))]:
                G += discount * reward
                discount *= self.gamma

            if len(states) > self.n:
                s_next = states[self.n]
                a_next = actions[self.n]
                G+= discount * self.Q[s_next, a_next]

            #TD error
            td_error = G - self.Q[s0, a0]
            self.Q[s0, a0] += self.alpha * td_error

            states.pop(0)
            actions.pop(0)
            rewards.pop(0)


    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for _ in range(n_episodes):
            env.reset()
            s = env.state()
            done = env.done()

            states = []
            actions = []
            rewards = []

            ep_return = 0
            max_steps = 1000
            steps = 0

            while not done and steps < max_steps:
                a = self.select_action(s)
                r = env.step(a)
                sp = env.state()
                done = env.done()

                states.append(s)
                actions.append(a)
                rewards.append(r)

                self.update(states.copy(), actions.copy(), rewards.copy(), done)
                s = sp
                ep_return += r
                steps += 1

            while len(states) > 0:
                self.update(states, actions, rewards, done=True)

            episode_returns.append(ep_return)

        return episode_returns