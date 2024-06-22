from network import Model
from game import Game
from train import PPOTrainer
from player import NNAgent, EVAgent

import matplotlib.pyplot as plt
import numpy as np
import torch

NUM_GAMES = 1000
NUM_TRIALS = 100
DEVICE = 'cpu'

trained_results = []
non_trained_results = []
input_space = 10
win_threshold = .55

action_space = list(range(3, 19)) #place a bid, place an ask
train_model = Model(input_space, len(action_space)) 
prev_best = Model(input_space, len(action_space))

for trial in range(NUM_TRIALS):

    # Self play
    training_data = []
    print(f'TRIAL {trial + 1}')
    for _ in range(NUM_GAMES):
        # print(f'GAME {_ + 1}')
        p1 = NNAgent(train_model)
        p2 = NNAgent(train_model)
        game = Game(p1, p2)
        training_data += game.play()
        for i, p in enumerate(game.players):
            profit = p.final_profit(game.cards_sum)
            # print(f'Player {i}: {profit}')

    permute_idxs = np.random.permutation(len(training_data[0]))

    # Policy data
    obs = torch.tensor(training_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
    acts = torch.tensor(training_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
    advantages = torch.tensor(training_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
    act_log_probs = torch.tensor(training_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

    # Value data
    returns = torch.tensor(training_data[2], dtype=torch.float32, device=DEVICE)

    # Train
    trainer = PPOTrainer(train_model)
    trainer.train_policy(obs, acts, act_log_probs, advantages)
    trainer.train_value(obs, returns)

    # Play previous benchmark
    win_count = 0
    for _ in range(NUM_GAMES):
        p1 = NNAgent(train_model)
        p2 = EVAgent()
        game = Game(p1, p2)
        game.play(train=False)
        if game.players[0].final_profit(game.cards_sum) > game.players[1].final_profit(game.cards_sum):
            win_count += 1
    print(f'Trained model won {win_count * 100 / NUM_GAMES}% of the games')
    if win_count / NUM_GAMES >= win_threshold:
        state_dict = train_model.state_dict()
        prev_best = Model(input_space, len(action_space))
        prev_best.load_state_dict(state_dict)


