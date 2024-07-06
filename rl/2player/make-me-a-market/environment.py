import random
import numpy as np

class Environment:
    def __init__(self):
        self.round = 0
        self.mean = 45
        self.std = 15
        self.real_sum = np.random.normal(self.mean, self.std)
        self.real_sum = round(self.real_sum)
        # self.real_sum = 60
        self.real_sum = max(self.real_sum, 5)
        self.real_sum = min(self.real_sum, 85)
        self.taker_pnl = 0
        self.taker_pos = 0
        self.guesses = []
        self.pos_history = []

    def reset(self):
        self.__init__()
        return [self.mean, self.std] + self.guesses + [0] * (5 - self.round) + [self.taker_pos]

    def step(self, middle):
        # defining trader
        # making sure that the market 

        middle = round(middle)
        
        if middle < 5 - self.round:
            middle = 5 - self.round
        elif middle > 85 + self.round:
            middle = 85 + self.round
        self.guesses.append(middle)
        
        if middle < self.real_sum: # ask is too low, dude will buy
            self.taker_pnl += self.real_sum - (middle + (5 - self.round))
            self.taker_pos += 1
        elif middle > self.real_sum: # bid is too high, dude will sell
            self.taker_pnl += (middle - (5 - self.round)) - self.real_sum
            self.taker_pos -= 1
        else:
            if self.taker_pos > 0:
                self.taker_pnl += ((middle - (5 - self.round)) - self.real_sum) * self.taker_pos
            elif self.taker_pos < 0:
                self.taker_pnl += (self.real_sum - (middle + (5 - self.round))) * self.taker_pos
            else:
                self.taker_pnl -= (5 - self.round) * 2
            self.taker_pos = 0
        self.pos_history.append(self.taker_pos)

        self.round += 1
        obs = [self.mean, self.std] + self.guesses + [0] * (5 - self.round) + [self.taker_pos]

        if self.round == 5:
            if self.taker_pos > 0:
                self.taker_pnl += self.taker_pos * self.real_sum
            elif self.taker_pos < 0:
                self.taker_pnl -= self.taker_pos * self.real_sum
        
        reward = -self.taker_pnl

        if middle == self.real_sum:
            reward += 100

        if self.round == 5: # end of game
            return None, reward
        else:
            return obs, reward