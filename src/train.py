import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --------------------------
# 1. Dummy Dataset
# --------------------------
class DummyApneaDataset(Dataset):
    def __init__(self, num_samples=200, img_size=(1, 64, 64)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random spectrogram-like data
        x = np.random.randn(*self.img_size).astype(np.float32)
        y = np.random.randint(0, 2)  # 0 = normal, 1 = apnea
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

# --------------------------
# 2. Simple CNN Model
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes (apnea / normal)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# --------------------------
# 3. Training Loop
# --------------------------
def train_model(epochs=5, batch_size=16, lr=0.001):
    # Load dummy data
    dataset = DummyApneaDataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("✅ Training finished (with dummy data)!")

if __name__ == "__main__":
    train_model()
