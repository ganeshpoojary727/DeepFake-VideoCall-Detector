import torch
import torch.nn as nn


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        device
    ):
        self.validation_loader = validation_loader
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    def train_one_epoch(self):

     self.model.train()

     running_loss = 0.0
     correct = 0
     total = 0

     for features, labels in self.train_loader:

        features = features.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(features)

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loss = self.criterion(outputs, labels)

        loss.backward()

        self.optimizer.step()

        running_loss += loss.item()

     epoch_loss = running_loss / len(self.train_loader)

     epoch_accuracy = 100 * correct / total

     return epoch_loss, epoch_accuracy
    def validate(self):

     self.model.eval()

     running_loss = 0.0

     correct = 0
     total = 0

     with torch.no_grad():

        for features, labels in self.validation_loader:

            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

     epoch_loss = running_loss / len(self.validation_loader)

     epoch_accuracy = 100 * correct / total

     return epoch_loss, epoch_accuracy