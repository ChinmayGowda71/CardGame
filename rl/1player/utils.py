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
    train_data = [[], [], [], [], []] # obs, bid_act, ask_act, reward, bid_val, ask_val, bid_act_log_prob, ask_act_log_prob
    obs = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        obs = np.append(obs, [1])

        bid_logits, bid_val = model(torch.tensor([obs], dtype=torch.float32, device=DEVICE))

        bid_act_distribution = Categorical(logits=bid_logits)
        bid_act = bid_act_distribution.sample()
        bid_act_log_prob = bid_act_distribution.log_prob(bid_act).item()
        bid_act, bid_val = bid_act.item(), bid_val.item() + 3

        obs[-1] = 0
        ask_logits, ask_val = model(torch.tensor([obs], dtype=torch.float32, device=DEVICE))
        ask_logits = ask_logits[0]
        for i in range(len(ask_logits)):
            if i + 4 <= bid_val:
                ask_logits[i] = -float('inf')
            else:
                break

        ask_act_distribution = Categorical(logits=ask_logits)
        ask_act = ask_act_distribution.sample()
        ask_act_log_prob = ask_act_distribution.log_prob(ask_act).item()
        ask_act, ask_val = ask_act.item(), ask_val.item() + 4

        next_obs, reward = env.step([bid_act, ask_act])

        for i, item in enumerate((obs, [bid_act, ask_act], reward, [bid_val, ask_val], [bid_act_log_prob, ask_act_log_prob])):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if obs == None:
            break

    train_data = [np.asarray(x) for x in train_data]

    ### Do train data filtering
    train_data[3][0] = calculate_advantages(train_data[2], train_data[3][0])
    train_data[3][1] = calculate_advantages(train_data[2], train_data[3][1])

    return train_data, ep_reward