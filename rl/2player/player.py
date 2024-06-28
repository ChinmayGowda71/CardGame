import numpy as np
import torch
from torch.distributions.categorical import Categorical
import math

DEVICE='cpu'

class Player:
    def __init__(self):
        self.state = {
            'card': 0,
            'money': 0,
            'shares': 0
        }

    def encode_state(self):
        return [self.state['card'], self.state['money'], self.state['shares']]
    
    def final_profit(self, card_value):
        return self.state["money"] + self.state["shares"] * card_value
    
class NNAgent(Player):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.trainable = True

    def inference(self, data):
        policy_logits, value = self.model(torch.tensor(data, dtype=torch.float32, device=DEVICE))
        policy_logits = policy_logits.clone().detach() if isinstance(policy_logits, torch.Tensor) else torch.tensor(policy_logits, dtype=torch.float32, device=DEVICE).clone().detach()
        if data[-1] != 0:
            for i in range(len(policy_logits)):
                if i + 4 <= data[-1]:
                    policy_logits[i] = -float('inf')
        dist = Categorical(logits=policy_logits)
        return dist, value.item()

    def play(self, market_state, bid=True, bid_val=None, round=0):
        state = np.array(super().encode_state() + market_state + ([1, 0] if bid else [0, bid_val]), dtype=np.float32)
        dist, value = self.inference(state)
        move = dist.sample()
        return state, move.item(), value, dist.log_prob(move).item(), False
    
class EVAgent(Player):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def play(self, market_state, bid=True, bid_val=None, round=0):
        visible_sum = market_state[-1]
        ev = (2 - round) * (55 - visible_sum - self.state['card']) / (9 - round) + round * visible_sum
        if int(ev) != ev:
            bid_val, ask_val = math.floor(ev), math.ceil(ev)
        else:
            bid_val, ask_val = ev - 1, ev + 1
        if bid:
            return 0, bid_val, 0, 0, True
        else:
            return 0, ask_val, 0, 0, True

    