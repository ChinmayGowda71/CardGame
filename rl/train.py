import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    def __init__(self, model, data, policy, scores):
        self.epochs = 100
        self.model = model
        self.x = torch.tensor(data)
        self.policy_y = torch.tensor(policy)
        self.value_y = torch.tensor(scores)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.policy_optimizer = optim.Adam(list(model.shared_layers.parameters()) + list(model.policy_layers.parameters()), lr=0.1)
        self.value_optimizer = optim.Adam(list(model.shared_layers.parameters()) + list(model.value_layers.parameters()), lr=0.1)

    def train_policy(self):
        dataset = TensorDataset(self.x, self.policy_y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        running = []
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                # Zero the parameter gradients
                self.policy_optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(inputs)

                # Compute loss
                loss = self.cross_entropy(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.policy_optimizer.step()

                # Print statistics
                running_loss += loss.item()
                running.append(loss.item())
            # if epoch % 5 == 0:
            #     print(f"Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
            #     running_loss = 0.0


    def train_value(self):
        dataset = TensorDataset(self.x, self.value_y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                # Zero the parameter gradients
                self.value_optimizer.zero_grad()

                # Forward pass
                _, outputs = self.model(inputs)
                outputs = torch.reshape(outputs, (-1,))

                # Compute loss
                loss = self.mse(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.value_optimizer.step()

                # Print statistics
                running_loss += loss.item()
            # if epoch % 5 == 0:
            #     print(f"Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
            #     running_loss = 0.0