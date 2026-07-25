import torch
import torch.nn as nn
import torch.optim as optim

from app.ai.datasets.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)
from app.ai.models.cnn_model import DeepFakeCNN
from app.ai.training.trainer import Trainer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device : {device}")

train_loader = create_train_dataloader()

validation_loader = create_validation_dataloader()

model = DeepFakeCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

num_epochs = 10

for epoch in range(num_epochs):

    loss,accuracy = trainer.train_one_epoch()
    val_loss, val_accuracy = trainer.validate()

    print(
    f"Epoch [{epoch + 1}/{num_epochs}] | "
    f"Train Loss: {loss:.4f} | "
    f"Train Acc: {accuracy:.2f}% | "
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_accuracy:.2f}%"
)

print(f"Training Loss : {loss:.4f}")