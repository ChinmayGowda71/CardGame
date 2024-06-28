from utils import *

import random
import numpy as np

class Game:

    num_players = 2
    rounds = 2

    def __init__(self, player1, player2, starting_money=50, starting_shares=0):
        self.market = [
            {
                'bid': 0,
                'ask': 0,
                'prev_bid_ask': [0, 0]
            },
            {
                'bid': 0,
                'ask': 0,
                'prev_bid_ask': [0, 0]
            }
        ]
        self.cards = random.sample(range(1, 11), self.num_players + 2)

        self.players = [player1, player2]
        for i in range(self.num_players):
            state = {
                'card': self.cards[i], 
                'money': 0, 
                'shares': starting_shares
            }
            self.players[i].state = state

        self.cards_sum = self.cards[-2] + self.cards[-1]
        self.visible_sum = 0
        self.starting_money = starting_money
        self.starting_shares = starting_shares
        self.r = 0
        self.training_data = [
            {
                'bid': [[], [], [], [], []],
                'ask': [[], [], [], [], []]
            },
            {
                'bid': [[], [], [], [], []],
                'ask': [[], [], [], [], []]
            }
        ]

    def encode_market_for_player(self, player_i):
        state = []
        state += self.market[player_i]['prev_bid_ask']
        for i in range(self.num_players):
            if i == player_i:
                continue
            state += self.market[i]['prev_bid_ask']

        return state + [self.visible_sum]                

    def play(self):
        for r in range(self.rounds):
            if self.r != 0:
                self.visible_sum += self.cards[-self.r]
            bid_info = [[] for i in self.players]
            ask_info = [[] for i in self.players]
            for p, player in enumerate(self.players):
                bid_state, bid, bid_pred_value, bid_act_log_prob, bid_adj = player.play(self.encode_market_for_player(p), bid=True, round=r)
                ask_state, ask, ask_pred_value, ask_act_log_prob, ask_adj = player.play(self.encode_market_for_player(p), bid=False, bid_val=bid, round=r)
                if player.trainable:
                    bid_info[p] = [bid_state, bid, bid_pred_value, bid_act_log_prob]
                    ask_info[p] = [ask_state, ask, ask_pred_value, ask_act_log_prob]
                if not bid_adj:
                    bid += 3
                if not ask_adj:
                    ask += 4
                self.market[p]["bid"] = bid
                self.market[p]["ask"] = ask
                self.market[p]['prev_bid_ask'] = [self.market[p]["bid"], self.market[p]["ask"]]

            cur_money = np.array([p.final_profit(self.cards_sum) for p in self.players])

            if self.market[1]["bid"] >= abs(self.market[0]["ask"]) and self.market[0]["ask"] != 0 and self.market[1]["bid"] != 0:
                self.players[1].state['money'] -= abs(self.market[0]["ask"])
                self.players[1].state["shares"] += 1

                self.players[0].state['money'] += abs(self.market[0]["ask"])
                self.players[0].state["shares"] -= 1

            if self.market[0]["bid"] >= abs(self.market[1]["ask"]) and self.market[1]["ask"] != 0 and self.market[0]["bid"] != 0:
                self.players[0].state["money"] -= abs(self.market[1]["ask"])
                self.players[0].state["shares"] += 1

                self.players[1].state["money"] += abs(self.market[1]["ask"])
                self.players[1].state["shares"] -= 1

            new_money = np.array([p.final_profit(self.cards_sum) for p in self.players])
            profits = new_money - cur_money
            self.r += 1

            for p in range(self.num_players):
                if self.players[p].trainable:
                    for i, item in enumerate(bid_info[p][:2] + [profits[p]] + bid_info[p][2:]):
                        self.training_data[p]['bid'][i].append(item)
                    for i, item in enumerate(ask_info[p][:2] + [profits[p]] + ask_info[p][2:]):
                        self.training_data[p]['ask'][i].append(item)
                    
        training_data = [np.empty((0, 10)), np.empty(0,), np.empty(0,), np.empty(0,), np.empty(0,)]
        for p in range(self.num_players):
            if self.players[p].trainable:
                for act in ['bid', 'ask']:
                    self.training_data[p][act] = [np.asarray(x) for x in self.training_data[p][act]]
                    self.training_data[p][act][3] = calculate_advantages(self.training_data[p][act][2], self.training_data[p][act][3])
                    self.training_data[p][act][2] = discount_rewards(self.training_data[p][act][2])
                    for i in range(len(training_data)):
                        training_data[i] = np.append(training_data[i], self.training_data[p][act][i], axis=0)
    
        return training_data