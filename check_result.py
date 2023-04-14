import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2


if __name__ == '__main__':


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    env_name = 'simple_tag_v2'
    episode_num = 30000
    result_dir = './results/simple_tag_v2/1/'
    f = open(result_dir+'rewards.pkl', 'rb')
    episode_rewards = pickle.load(f)['rewards']
    f.close()
    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
