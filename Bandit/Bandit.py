import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List


class Bandit:
    '''
    k-armed bandit problem in stationary case. Each Bandit instance generate a normal distribuiton for action reward means.
    Each action rewards also has normal distribuiton with the same mean and stdev.
    Play function(s) return a reward based on generated distiribution
    '''

    def __init__(self, arms: int, mean: float, stdev: float, stationary: bool = True, fn_step_size: Callable = None) -> None:
        assert arms >= 1
        # generate reward means distribuition for each of k actions
        self.arms = arms
        self.reward_means = np.random.normal(loc=mean, scale=stdev, size=arms)
        self.mean = mean
        self.stdev = stdev
        self.stationary = stationary
        if fn_step_size is None:
            self.fn_step_size = self._step_average
        else:
            self.fn_step_size = fn_step_size
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.arms)
        self.nQ = np.zeros(self.arms)

    def get_Q(self) -> np.ndarray:
        return self.Q

    def get_true_Q(self) -> np.ndarray:
        return self.reward_means

    def _step_average(self, n: int) -> float:
        return 1/n

    def _random_walk_true_Q(self) -> None:
        walks = np.random.normal(loc=0, scale=0.01, size=self.arms)
        self.reward_means += walks

    def _update_Q(self, action: int, reward: float) -> None:
        if not self.stationary:
            self._random_walk_true_Q()

        self.nQ[action] = self.nQ[action] + 1
        self.Q[action] = self.Q[action] + \
            (reward - self.Q[action]) * self.fn_step_size(self.nQ[action])

    def play(self, action: int) -> float:
        assert 0 <= action < self.arms
        # generate reward distribution for an action with the same parameters
        reward = np.random.normal(
            loc=self.reward_means[action], scale=self.stdev, size=1)
        self._update_Q(action, reward)
        return reward

    def play_best(self) -> float:
        action = np.argmax(self.get_Q())
        return self.play(action)

    def play_random(self) -> float:
        action = np.random.choice(np.arange(self.arms, dtype=int))
        return self.play(action)


def estimate_reward(bandit: Bandit, arms: int, eps: float) -> float:
    reward = 0
    while True:
        # Random action is taken with p = eps, greedy action is taken otherwise
        if 0 == np.random.choice([0, 1], p=[eps, 1 - eps]):
            reward = bandit.play_random()
        else:
            reward = bandit.play_best()
        yield reward


def run_bandits_common(n_bandits: int, steps: int, arms: int, mean: float, stdev: float,
                       all_eps: List[float], stationary: bool, plot_title: str,
                       fn_step_size: Callable = None) -> None:
    all_bandits = []
    # Initiliaze bandits once. We will re-use same bandits for different values of eps
    for _ in range(n_bandits):
        all_bandits.append(
            Bandit(arms, mean, stdev, stationary=stationary, fn_step_size=fn_step_size))

    # Find best reward assuming we taking best actions every time
    all_true_q = []
    for bandit in all_bandits:
        all_true_q.append(np.max(bandit.get_true_Q()))
    best_reward = np.mean(all_true_q)

    # Estimate Q for different values of eps (eps-greedy)
    plt.figure(figsize=(15, 10))
    for eps in all_eps:
        all_rewards = []
        for bandit in all_bandits:
            # reset reward estimations Q
            bandit.reset()
            bandit_rewards = []
            gen_bandit = estimate_reward(bandit, arms, eps)
            for _ in range(steps):
                bandit_rewards.append(next(gen_bandit))
            all_rewards.append(bandit_rewards)
        # Get mean across all bandits for each step
        steps_mean = np.mean(all_rewards, axis=0)
        plt.plot(np.arange(steps), steps_mean, '-', label=("eps=%.2f" % eps))

    plt.title(plot_title)
    plt.plot(np.full(steps, best_reward), label="Stationary optimal reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward mean")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    pass
