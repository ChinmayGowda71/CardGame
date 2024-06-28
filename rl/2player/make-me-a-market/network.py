from torch import nn
import torch

# Policy and value model
class Model(nn.Module):
  def __init__(self, obs_space_size, action_space):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 512),
        nn.ReLU(),
        nn.LayerNorm(512), 
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.LayerNorm(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.LayerNorm(128)
    )
    
    self.policy_layers = nn.Sequential(
        nn.Linear(128, action_space)
    )
    
    self.value_layers = nn.Sequential(
        nn.Linear(128, 1),
        nn.Tanh()
    )
    
    self._initialize_weights()
    
  def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value