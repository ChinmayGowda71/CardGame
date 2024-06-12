from player import Player

import random

class Game:

    num_players = 2
    rounds = 3

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
        # self.market = [-1, -1, -1] # index of player who made last trade, bid/ask (1 for bid, 0 for ask), price
        cards = random.sample(range(1, 11), self.num_players + 2)

        self.players = []
        for i in range(self.num_players):
            if control != None and i == 1:
                player = Player(control, action_space)
            else:
                player = Player(model, action_space)
            state = {
                'card': cards[i], 
                'money': starting_money, 
                'shares': starting_shares
            }
            player.state = state
            self.players.append(player)

        self.cards_sum = cards[-2] + cards[-1]
        self.starting_money = starting_money
        self.starting_shares = starting_shares

    def encode_market_for_player(self, player_i):
        state = []
        state += [self.market[player_i]['bid'], self.market[player_i]['bid']] + self.market[player_i]['prev_bid_ask']
        for i in range(self.num_players):
            if i == player_i:
                continue
            state += [self.market[i]['bid'], self.market[i]['bid']] + self.market[i]['prev_bid_ask']

        return state

    def play(self):
        for _ in range(self.rounds):
            for p, player in enumerate(self.players):
                bid = player.play(self.encode_market_for_player(p), bid=True)
                ask = player.play(self.encode_market_for_player(p), bid=False)
                self.market[p]["bid"] = bid
                self.market[p]["ask"] = ask

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

            for p in range(self.num_players):
                self.market[p]['prev_bid_ask'] = [self.market[p]["bid"], self.market[p]["ask"]]


def custom_minmax_scaler(data, feature_range=(-1, 1)):
    min_val, max_val = 0, 100
    range_min, range_max = feature_range
    scale = (range_max - range_min) / (max_val - min_val)
    min_scaled = range_min - min_val * scale
    scaled_data = data * scale + min_scaled
    return scaled_data