from network import Model
from game import Game
from train import PPOTrainer

import matplotlib.pyplot as plt
import numpy as np
import torch

NUM_GAMES = 10
NUM_TRIALS = 10
DEVICE = 'cpu'

trained_results = []
non_trained_results = []
input_space = 9
action_space = list(range(3, 19)) #place a bid, place an ask
train_model = Model(input_space, len(action_space)) # 
control_model = Model(input_space, len(action_space))
for trial in range(NUM_TRIALS):

    training_data = []
    print(f'TRIAL {trial + 1}')
    for _ in range(NUM_GAMES):
        game = Game(train_model, action_space)
        training_data += game.play()
        for i, p in enumerate(game.players):
            profit = p.final_profit(game.cards_sum)
            print(f'Player {i}: {profit}')

    permute_idxs = np.random.permutation(len(training_data[0]))

    # Policy data
    obs = torch.tensor(training_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
    acts = torch.tensor(training_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
    advantages = torch.tensor(training_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
    act_log_probs = torch.tensor(training_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

    # Value data
    returns = torch.tensor(training_data[2], dtype=torch.float32, device=DEVICE)

    trainer = PPOTrainer(train_model)
    trainer.train_policy(obs, acts, act_log_probs, advantages)
    trainer.train_value(obs, returns)

    # # test the model against an untrained model with random weights
    # print('Playing against random model')
    # game = Game(train_model, action_space, control=control_model)
    # game.play()
    # for i, p in enumerate(game.players):
    #     profit = p.final_profit(game.cards_sum)
    #     print(f'Player {i + 1}: {profit}')
    # print()

# plt.plot(trained_results)
# plt.plot(non_trained_results)
# plt.show()