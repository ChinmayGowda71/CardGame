from node import Node

import torch
from torch.distributions.categorical import Categorical

class MCTS:
    def __init__(self, model, action_space):
        self.model = model
        self.iters = 25
        self.action_space = action_space

    def get_search_probabilities(self, node):
        logits = [c.N for c in node.children]
        if isinstance(logits, torch.Tensor):
            logits = logits.clone().detach()  # If logits is already a tensor
        else:
            logits = torch.tensor(logits, dtype=torch.float32)  # Convert to tensor if it's a list
        logits = Categorical(logits=logits)
        return logits.probs
    
    # def get_search_probabilities(self, node):
    #     logits = [c.N for c in node.children]
    #     logits = Categorical(logits=torch.tensor(logits, dtype=torch.float32).clone().detach())  # Fix this line
    #     return logits.probs

    def get_next_move(self, state):
        root = Node()
        root.state = state

        for _ in range(self.iters):
            node = root

            # 1. Selection
            while len(node.children) > 0:
                max_score = 0
                child_with_max_score = None
                for c in node.children:
                    if c.score() > max_score or child_with_max_score == None:
                        child_with_max_score = c
                        max_score = c.score()
                node = child_with_max_score
            
            # 2 and 3. Expansion and Rollout
            policy_logits, value = self.model(torch.tensor(state, dtype=torch.float32).clone().detach())
            policy_logits = policy_logits.clone().detach() if isinstance(policy_logits, torch.Tensor) else torch.tensor(policy_logits, dtype=torch.float32).clone().detach()
            probs = Categorical(logits=policy_logits).probs  # Fix this line
            # policy_logits, value = self.model(torch.tensor(state, dtype=torch.float32).clone().detach())  # Fix this line
            # value = torch.reshape(value, (-1,))
            # probs = Categorical(logits=torch.tensor(policy_logits, dtype=torch.float32).clone().detach()).probs
            # policy_logits, value = self.model(torch.tensor(state))
            # probs = Categorical(logits=torch.tensor(policy_logits)).probs
            # if bid:
            #     probs[len(probs) // 2 + 1:] = 0
            # else:
            #     probs[:len(probs) // 2] = 0
            # probs /= probs.sum()
            for i, a in enumerate(self.action_space):
                # update state
                if probs[i] == 0:
                    continue
                new_state = state

                new_node = Node()
                new_node.state = new_state
                new_node.parent = node
                new_node.P = policy_logits[i]
                new_node.action = a

                node.children.append(new_node)

            # 4. Backpropogation
            while node is not None:
                node.N += 1
                node.W += value
                node.Q = node.W / node.N
                node = node.parent

        return max(root.children, key=lambda c: c.N).action, self.get_search_probabilities(root)

             
            
            

