import torch
import torch.nn as nn


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        criterion,
        device
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    def train_one_epoch(self):

      self.model.train()

      running_loss = 0.0

      for features, labels in self.train_loader:

        # Move data to GPU/CPU
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Clear old gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(features)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        self.optimizer.step()

        # Add batch loss
        running_loss += loss.item()

      epoch_loss = running_loss / len(self.train_loader)

      return epoch_loss