import math
import random

class Environment:
    def __init__(self):
        self.state = [0, random.randint(1, 10)] # EV agent's EV
        self.ev_card_r1 = 0
        self.common_card = 0
        self.round = 0

    def reset(self):
        self.__init__()
        return self.state

    def step(self, action):

        profit = 0

        if self.round == 0:
            self.state[0] = 2 * ((55 - self.ev_card_r1) / 9)
        elif self.round == 1:
            self.state[0] = (55 - self.ev_card_r1 - self.common_card) / 8 + self.common_card
        else:
            return None
        

        if self.state[0] / math.floor(self.state[0]) == self.state[0]:
            ev_bid = math.floor(self.state[0])
            ev_ask = math.ceil(self.state[0])
        else:
            ev_bid = self.state[0] - 1
            ev_ask = self.state[0] + 1

        if action[0] >= ev_ask:
            model_money -= action[0]
            model_shares += 1
            
            ev_shares -= 1
            ev_money += action[0]

        if ev_bid >= action[1]:
            model_money += ev_bid
            model_shares -= 1

            ev_shares += 1
            ev_money -= ev_bid

        
        """Accepts a list containing the bid amount and ask amount in that order - one of them needs to be negated
        Should return the next state after executing the trades, as well as the profit after executing the trades"""
        self.round += 1
        return self.state, profit