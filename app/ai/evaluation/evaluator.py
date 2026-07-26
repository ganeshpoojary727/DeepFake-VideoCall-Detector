import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

class Evaluator:

    def __init__(
        self,
        model,
        test_loader,
        device
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):

     self.model.eval()

     correct = 0
     total = 0

     all_predictions = []
     all_labels = []

     with torch.no_grad():

        for features, labels in self.test_loader:

            # Move data to GPU/CPU
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(features)

            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

     accuracy = accuracy_score(all_labels, all_predictions)

     precision = precision_score(all_labels, all_predictions)

     recall = recall_score(all_labels, all_predictions)

     f1 = f1_score(all_labels, all_predictions)

     matrix = confusion_matrix(all_labels, all_predictions)

     report = classification_report(all_labels, all_predictions)

     return (
        accuracy,
        precision,
        recall,
        f1,
        matrix,
        report
    )