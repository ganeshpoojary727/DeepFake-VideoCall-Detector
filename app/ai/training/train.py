import torch
import torch.nn as nn
import torch.optim as optim

from app.ai.datasets.dataloader import create_dataloader
from app.ai.models.cnn_model import DeepFakeCNN
from app.ai.training.trainer import Trainer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device : {device}")

train_loader = create_dataloader()

model = DeepFakeCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

num_epochs = 10

for epoch in range(num_epochs):

    loss = trainer.train_one_epoch()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Training Loss: {loss:.4f}"
    )

print(f"Training Loss : {loss:.4f}")