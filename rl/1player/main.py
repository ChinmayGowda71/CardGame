from trainer import PPOTrainer
from network import Model
from utils import *
from environment import Environment

# Define training params
n_episodes = 200
print_freq = 20
model = Model(1 + 1, len(range(3, 19)))

ppo = PPOTrainer(
    model,
    policy_lr = 3e-4,
    value_lr = 1e-3,
    target_kl_div = 0.02,
    max_policy_train_iters = 40,
    value_train_iters = 40)

# Training loop
ep_rewards = []
for episode_idx in range(n_episodes):
  # Perform rollout
  env = Environment()
  train_data, reward = rollout(model, env)
  ep_rewards.append(reward)

  # Shuffle
  permute_idxs = np.random.permutation(len(train_data[0]))

  # Policy data
  obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
  acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
  advantages = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
  act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

  # Value data
  returns = discount_rewards(train_data[2])[permute_idxs]
  returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

  # Train model
  ppo.train_policy(obs, acts, act_log_probs, advantages)
  ppo.train_value(obs, returns)

  if (episode_idx + 1) % print_freq == 0:
    print('Episode {} | Avg Reward {:.1f}'.format(
        episode_idx + 1, np.mean(ep_rewards[-print_freq:])))