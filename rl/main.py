from network import Model
from game import Game
from train import Trainer

import matplotlib.pyplot as plt
import numpy as np

NUM_GAMES = 10
NUM_TRIALS = 10
trained_results = []
non_trained_results = []
action_space = list(range(3, 20)) + [0] #place a bid, place an ask
train_model = Model(12, len(action_space)) # 
control_model = Model(12, len(action_space))
for trial in range(NUM_TRIALS):

    training_data = []
    search_probs = []
    scores = []
    print(f'TRIAL {trial + 1}')
    profits1 = []
    profits2 = []
    for _ in range(NUM_GAMES):
        game = Game(train_model, action_space)
        game.play()
        for i, p in enumerate(game.players):
            profit = p.final_profit(game.cards_sum)
            # print(f'Player {i + 1}: {profit}')
            training_data += p.states_actions
            search_probs += p.search_probs

            # scores += [custom_minmax_scaler(profit)] * len(p.states_actions)
            scores += [1 if profit > game.starting_money else -1] * len(p.states_actions)
            if i == 0:
                profits1.append(profit)
            else:
                profits2.append(profit)

    trained_results.append(sum(profits1) / len(profits1))
    non_trained_results.append(sum(profits2) / len(profits2))
    print(sum(profits1) / len(profits1), sum(profits2) / len(profits2))
    print()
    trainer = Trainer(train_model, np.array(training_data, dtype=np.float32), np.array(search_probs, dtype=np.float32), np.array(scores, dtype=np.float32))
    trainer.train_policy()
    trainer.train_value()

    # test the model against an untrained model with random weights
    print('Playing against random model')
    game = Game(train_model, action_space, control=control_model)
    game.play()
    for i, p in enumerate(game.players):
        profit = p.final_profit(game.cards_sum)
        print(f'Player {i + 1}: {profit}')
    print()

plt.plot(trained_results)
plt.plot(non_trained_results)
plt.show()