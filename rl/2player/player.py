from mcts import MCTS

import numpy as np
import torch
from torch.distributions.categorical import Categorical

class Player:
    def __init__(self, model, player_i):
        self.player_i = player_i
        self.model = model
        # self.mcts = MCTS(model)
        self.state = {
            'card': 0,
            'money': 0,
            'shares': 0
        } #card, money, shares, average cost
        self.search_probs = []

    def encode_state(self):
        return [self.state['card'], self.state['money'], self.state['shares']]
    
    def inference(self, data, bid_val):
        policy_logits, value = self.model(torch.tensor(data, dtype=torch.float32))
        policy_logits = policy_logits.clone().detach() if isinstance(policy_logits, torch.Tensor) else torch.tensor(policy_logits, dtype=torch.float32).clone().detach()
        if bid_val != None:
            for i in range(len(policy_logits)):
                if i + 4 <= bid_val:
                    policy_logits[i] = -float('inf')
        dist = Categorical(logits=policy_logits)
        move = dist.sample()
        return dist, value.item()

    def play(self, market_state, bid=True, bid_val=None):
        state = np.array(self.encode_state() + market_state + [1 if bid else 0], dtype=np.float32)
        dist, value = self.inference(state, bid_val)
        move = dist.sample()
        # (obs, [bid_act, ask_act], reward, [bid_val, ask_val], [bid_act_log_prob, ask_act_log_prob])
        return state, move.item(), value, dist.log_prob(move).item()
    
    def final_profit(self, card_value):
        return self.state["money"] + self.state["shares"] * card_value