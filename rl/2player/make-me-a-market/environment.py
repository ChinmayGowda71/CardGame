import random
import numpy as np

class Environment:
    def __init__(self):
        self.round = 0
        self.mean = 45
        self.std = 15
        self.real_sum = np.random.normal(self.mean, self.std)
        self.real_sum = 60
        self.taker_pnl = 0
        self.taker_pos = 0
        self.guesses = []

    def reset(self):
        self.__init__()
        middle = 45
        return self.step(middle)

    def step(self, middle):
        # defining trader
        # making sure that the market 

        middle = round(middle)
        
        if middle < 5 - self.round:
            middle = 5 - self.round
        elif middle > 85 + self.round:
            middle = 85 + self.round
        self.guesses.append(middle)
        
        if middle + (5 - self.round) < self.real_sum: # ask is too low, dude will buy
            self.taker_pnl += self.real_sum - (middle + (5 - self.round))
            self.taker_pos += 1
        elif middle - (5 - self.round) > self.real_sum: # bid is too high, dude will sell
            self.taker_pnl += (middle - (5 - self.round)) - self.real_sum
            self.taker_pos -= 1
        else:
            self.taker_pnl -= (5 - self.round) * 2
            self.taker_pos = 0

        self.round += 1
        obs = [self.mean, self.std] + self.guesses + [0] * (5 - self.round) + [self.taker_pos]

        if self.round == 1:
            return obs

        if self.round == 5:
            if self.taker_pos > 0:
                self.taker_pnl += self.taker_pos * self.real_sum
            elif self.taker_pos < 0:
                self.taker_pnl -= self.taker_pos * self.real_sum
        
        # diff = self.guesses[-1] - self.guesses[-2]

        # reward = diff * self.taker_pos
        # reward = 1
        # if abs(self.taker_pos) >= 2:
        #     return None, reward

        reward = -self.taker_pnl

        if self.round == 5: # end of game
            return None, reward
        else:
            return obs, reward