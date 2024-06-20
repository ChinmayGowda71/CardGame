from player import Player
from utils import *

import random
import numpy as np

class Game:

    num_players = 2
    rounds = 2

    def __init__(self, model, action_space, starting_money=50, starting_shares=0, control=None):
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

        self.players = []
        for i in range(self.num_players):
            if control != None and i == 1:
                player = Player(control, action_space)
            else:
                player = Player(model, action_space)
            state = {
                'card': self.cards[i], 
                'money': starting_money, 
                'shares': starting_shares
            }
            player.state = state
            self.players.append(player)

        self.cards_sum = self.cards[-2] + self.cards[-1]
        self.visible_sum = 0
        self.starting_money = starting_money
        self.starting_shares = starting_shares
        self.r = 0
        self.training_data = [[], [], [], [], []]

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
                bid_state, bid, bid_pred_value, bid_act_log_prob = player.play(self.encode_market_for_player(p), bid=True)
                ask_state, ask, ask_pred_value, ask_act_log_prob = player.play(self.encode_market_for_player(p), bid=False, bid_val=bid)
                bid_info[p] = [bid_state, bid, bid_pred_value, bid_act_log_prob]
                ask_info[p] = [ask_state, ask, ask_pred_value, ask_act_log_prob]
                
                bid += 3
                ask += 4
                self.market[p]["bid"] = bid
                self.market[p]["ask"] = ask

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
                self.market[p]['prev_bid_ask'] = [self.market[p]["bid"], self.market[p]["ask"]]
                for i, item in enumerate(bid_info[p][:2] + [profits[p]] + bid_info[p][2:]):
                    self.training_data[i].append(item)
                for i, item in enumerate(ask_info[p][:2] + [profits[p]] + ask_info[p][2:]):
                    self.training_data[i].append(item)

        self.training_data = [np.asarray(x) for x in self.training_data]
        self.training_data[3] = calculate_advantages(self.training_data[2], self.training_data[3])
        self.training_data[2] = discount_rewards(self.training_data[2])
        return self.training_data