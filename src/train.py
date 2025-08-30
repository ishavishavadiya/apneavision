import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import dataset
from src.data_loader import ApneaSpectrogramDataset


# --------------------------
# 1. CNN Model
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),   # adjust if spectrogram size changes
            nn.ReLU(),
            nn.Linear(64, 2)              # 2 classes: apnea / non-apnea
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# --------------------------
# 2. Training Loop
# --------------------------
def train_model(
    epochs=10,
    batch_size=16,
    lr=0.001,
    checkpoint_dir="checkpoints",
    processed_folder="data/processed"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load dataset
    dataset = ApneaSpectrogramDataset(processed_folder=processed_folder)

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_acc = 100 * correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")

    print("✅ Training finished!")


if __name__ == "__main__":
    train_model()
