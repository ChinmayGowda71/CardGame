from mcts import MCTS
import numpy as np

class Player:
    def __init__(self, model, action_space):
        self.mcts = MCTS(model, action_space)
        self.state = {
            'card': 0,
            'money': 0,
            'shares': 0
        } #card, money, shares, average cost
        self.states_actions = []
        self.search_probs = []

    def encode_state(self):
        return [self.state['card'], self.state['money'], self.state['shares']]

    def play(self, market_state, bid=True):
        state = self.encode_state()
        move, probs = self.mcts.get_next_move(np.array(state + market_state + [1 if bid else 0], dtype=np.float32))
        self.states_actions.append(np.array(state + market_state + [1 if bid else 0], dtype=np.float32))
        self.search_probs.append(probs)
        return move
    
    def final_profit(self, card_value):
        return self.state["money"] + self.state["shares"] * card_value