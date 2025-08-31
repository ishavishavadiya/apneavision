import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Dataset
# ---------------------------
class SleepEpochDataset(Dataset):
    def __init__(self, data_dir, max_samples=None, use_fp16=True):
        epochs_path = os.path.join(data_dir, "epochs.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

        self.epochs = np.load(epochs_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        if max_samples:
            self.epochs = self.epochs[:max_samples]
            self.labels = self.labels[:max_samples]

        self.epochs = (self.epochs - self.epochs.min()) / (self.epochs.max() - self.epochs.min())
        dtype = np.float16 if use_fp16 else np.float32
        self.epochs = self.epochs.astype(dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.epochs[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# Models
# ---------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, channels, samples):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, channels * samples),
            nn.Tanh()
        )

        self.channels = channels
        self.samples = samples

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        out = self.model(x)
        return out.view(out.size(0), self.channels, self.samples)


class Discriminator(nn.Module):
    def __init__(self, num_classes, channels, samples):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, channels * samples)

        self.model = nn.Sequential(
            nn.Linear(channels * samples * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.channels = channels
        self.samples = samples

    def forward(self, x, labels):
        flat_x = x.view(x.size(0), -1)
        c = self.label_emb(labels)
        d_in = torch.cat([flat_x, c], dim=1)
        return self.model(d_in)


# ---------------------------
# Training Loop with Checkpointing
# ---------------------------
def train_cgan_for_class(data_dir, target_class, epochs=20, batch_size=16, device="cpu", max_samples=20000):
    dataset = SleepEpochDataset(data_dir, max_samples=max_samples, use_fp16=True)

    mask = dataset.labels == target_class
    dataset.epochs = dataset.epochs[mask]
    dataset.labels = dataset.labels[mask]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latent_dim = 100
    num_classes = 5
    channels, samples = dataset[0][0].shape

    generator = Generator(latent_dim, num_classes, channels, samples).to(device)
    discriminator = Discriminator(num_classes, channels, samples).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    checkpoint_dir = "apneavision/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    g_path = os.path.join(checkpoint_dir, f"G_class{target_class}.pth")
    d_path = os.path.join(checkpoint_dir, f"D_class{target_class}.pth")

    # Resume if checkpoint exists
    start_epoch = 0
    if os.path.exists(g_path) and os.path.exists(d_path):
        print(f"🔄 Resuming from checkpoint for class {target_class}")
        generator.load_state_dict(torch.load(g_path, map_location=device))
        discriminator.load_state_dict(torch.load(d_path, map_location=device))

    for epoch in range(start_epoch, epochs):
        for i, (real_x, labels) in enumerate(loader):
            real_x, labels = real_x.to(device), labels.to(device)
            bs = real_x.size(0)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            real_pred = discriminator(real_x, labels)
            d_real_loss = criterion(real_pred, valid)

            z = torch.randn(bs, latent_dim, device=device)
            fake_x = generator(z, labels)
            fake_pred = discriminator(fake_x.detach(), labels)
            d_fake_loss = criterion(fake_pred, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_x = generator(z, labels)
            gen_pred = discriminator(gen_x, labels)
            g_loss = criterion(gen_pred, valid)

            g_loss.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(loader)}] "
                      f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        # Save checkpoint at end of epoch
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
        print(f"💾 Saved checkpoint for class {target_class}, epoch {epoch+1}")

    print(f"✅ Finished training CGAN for class {target_class}")


if __name__ == "__main__":
    data_dir = "apneavision/data/epochs"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    for cls in range(5):
        train_cgan_for_class(
            data_dir,
            target_class=cls,
            epochs=5,
            batch_size=8,
            device=device,
            max_samples=5000
        )
