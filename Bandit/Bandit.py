from functools import partial
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import List, Dict, Tuple


class Bandit:
    '''
    k-armed bandit problem emulator. Each Bandit instance generate a normal distribuiton for action reward means.
    Each action rewards also has normal distribuiton with the same mean and stdev.
    The 'play' function returns a reward based on supported algorithms passed by 'configure' function
    '''

    def __init__(self, arms: int, mean: float, stdev: float) -> None:
        assert arms >= 1
        # generate reward means distribuition for each of k actions
        self.arms = arms
        self.reward_means = np.random.normal(loc=mean, scale=stdev, size=arms)
        self.mean = mean
        self.stdev = stdev
        self._configured = False

    def reset(self) -> None:
        '''
        Reset bandit state but keep the same reward distirbuiton
        '''
        self.Q = np.full(self.arms, self.init_Q, dtype=float)
        # number of times each action is selected
        self.nQ = np.zeros(self.arms, dtype=int)
        # rewards mean
        self.mean_R = 0.0
        # global step counter
        self.step = 0

    def _step_sample_average(self, n: int, action: int, params: Dict) -> float:
        return 1/n

    def _step_sample_constant(self, n: int, action: int, params: Dict) -> float:
        alpha = params['alpha']
        return alpha

    def _random_walk(self) -> None:
        '''
        Emulate non-stationarity of the problem
        '''
        self.reward_means += np.random.normal(loc=0, scale=0.05, size=self.arms)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _updater_inc_error_non_stationary(self, action: int, reward: float, params: Dict) -> None:
        self._random_walk()
        self._updater_inc_error_stationary(action, reward, params)

    def _updater_gradient_stationary(self, action: int, reward: float, params: Dict) -> None:
        alpha = self._step_size_fn(self.nQ[action], action, self._step_size_params)
        action_prob = self.softmax(self.Q)
        if self.mean_R == 0:
            self.mean_R = reward
        # update vectorized
        Q_action = self.Q[action] + alpha * (reward - self.mean_R) * (1 - action_prob[action])
        self.Q -= alpha * (reward - self.mean_R) * action_prob
        self.Q[action] = Q_action
        # new reward average
        self.mean_R = self.mean_R + (reward - self.mean_R)/self.step

    def _updater_inc_error_stationary(self, action: int, reward: float, params: Dict) -> None:
        self.nQ[action] = self.nQ[action] + 1
        self.Q[action] += (reward - self.Q[action]) * \
            self._step_size_fn(self.nQ[action], action, self._step_size_params)

    def _action_selector_softmax(self, params) -> int:
        return np.random.choice(self.arms, size=None, p=self.softmax(self.Q))

    def _action_selector_greedy(self, params) -> int:
        return np.argmax(self.Q)

    def _action_selector_eps_greedy(self, params) -> int:
        eps = params['eps']
        if 0 == np.random.choice([0, 1], p=[eps, 1 - eps]):
            action = np.random.choice(np.arange(self.arms, dtype=int))
        else:
            action = np.argmax(self.Q)
        return action

    def _action_selector_ucb(self, params) -> int:
        c = params['c']  # confidence level
        scores = np.zeros(self.arms)
        for i in range(self.arms):
            if self.nQ[i] == 0:
                return i
            scores[i] = self.Q[i] + c * np.sqrt(np.log(i + 1)/(self.nQ[i]))
        return np.argmax(scores)

    def _get_reward(self, action: int) -> float:
        return np.random.normal(loc=self.reward_means[action], scale=self.stdev, size=None)

    def play(self) -> float:
        if not self._configured:
            raise Exception("Bandit not configured. Call bandit.configure before playing.")
        self.step += 1
        action = self._action_selector_fn(self._action_selector_params)
        reward = self._get_reward(action)
        self._step_updater_fn(action, reward, self._step_updater_params)
        return reward

    def play_many(self, steps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        rewards = np.zeros(steps)
        best_rewards = np.zeros(steps)
        for i in range(steps):
            rewards[i] = self.play()
            best_rewards[i] = np.max(self.reward_means)
        return rewards, best_rewards

    def configure(self, *,
                  step_size: str = "sample_average", step_size_params: Dict = None,
                  action_selector: str = "greedy", action_selector_params: Dict = None,
                  step_updater: str = "inc_to_error_stationary", step_updater_params: Dict = None,
                  init_Q: float = 0.) -> None:

        self.init_Q = init_Q

        # make it Dictionary to simplify typing
        if action_selector_params is None:
            self._action_selector_params = dict()
        else:
            self._action_selector_params = action_selector_params

        # make it Dictionary to simplify typing
        if step_size_params is None:
            self._step_size_params = dict()
        else:
            self._step_size_params = step_size_params

        # make it Dictionary to simplify typing
        if step_updater_params is None:
            self._step_updater_params = dict()
        else:
            self._step_updater_params = step_updater_params

        # learning rate parameter applied by updaters
        if step_size == "sample_average":
            self._step_size_fn = self._step_sample_average
        elif step_size == "constant":
            self._step_size_fn = self._step_sample_constant

        # they way to select an action to play
        if action_selector == "greedy":
            self._action_selector_fn = self._action_selector_greedy
        elif action_selector == "eps_greedy":
            self._action_selector_fn = self._action_selector_eps_greedy
        elif action_selector == "ucb":
            self._action_selector_fn = self._action_selector_ucb
        elif action_selector == "softmax":
            self._action_selector_fn = self._action_selector_softmax

        # the way to update weights (Q)
        if step_updater == "inc_to_error_stationary":
            self._step_updater_fn = self._updater_inc_error_stationary
        elif step_updater == "inc_to_error_non_stationary":
            self._step_updater_fn = self._updater_inc_error_non_stationary
        elif step_updater == "gradient_stationary":
            self._step_updater_fn = self._updater_gradient_stationary

        self.reset()
        self._configured = True

if __name__ == '__main__':
    pass
