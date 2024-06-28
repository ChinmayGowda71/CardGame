import numpy as np
import torch
from torch.distributions.categorical import Categorical

DEVICE = 'cpu'

def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_advantages(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    advantages = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        advantages.append(deltas[i] + decay * gamma * advantages[-1])

    return np.array(advantages[::-1])

def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], []] # obs, middle, reward, pred_val, log_prob
    obs = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor([obs], dtype=torch.float32, device=DEVICE))
        logits = logits[0]
        dist = Categorical(logits=logits)
        pred_val = dist.sample()
        log_prob = dist.log_prob(pred_val).item()

        next_obs, reward = env.step(pred_val.item())

        for i, item in enumerate((obs, pred_val, reward, val.item(), log_prob)):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if obs == None:
            break

    train_data = [np.asarray(x) for x in train_data]

    ### Do train data filtering
    train_data[3] = calculate_advantages(train_data[2], train_data[3])

    return train_data, ep_reward