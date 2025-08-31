import os
import numpy as np
import torch
from cgan import Generator  # Make sure this points to your cgan.py Generator
import argparse

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "apneavision/data/epochs"
CHECKPOINT_DIR = "apneavision/checkpoints/synthetic"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8          # smaller batch size for Mac M4
LATENT_DIM = 100
TOTAL_SAMPLES_PER_CLASS = {
    0: 20196,  # target total samples for class balancing
    1: 20196,
    2: 42523,  # already the largest class, optional to skip
    3: 20196,
    4: 20196
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Load Generator for a class
# -----------------------------
def load_generator_for_class(target_class):
    channels, samples = 8, 6000  # from your dataset
    gen = Generator(latent_dim=LATENT_DIM, num_classes=5, channels=channels, samples=samples).to(DEVICE)
    ckpt_path = f"apneavision/checkpoints/G_class{target_class}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Generator checkpoint missing for class {target_class}: {ckpt_path}")
    gen.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    gen.eval()
    return gen

# -----------------------------
# Generate Synthetic Samples
# -----------------------------
def generate_synthetic_for_class(target_class, total_samples):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"class{target_class}.npz")
    synth_data, synth_labels = [], []
    start_idx = 0

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        synth_data = [ckpt["x"]]
        synth_labels = [ckpt["y"]]
        start_idx = ckpt["x"].shape[0]
        print(f"🔄 Resuming class {target_class} from index {start_idx}")

    if start_idx >= total_samples:
        print(f"✅ Class {target_class} already fully generated ({start_idx} samples)")
        return np.concatenate(synth_data, axis=0), np.concatenate(synth_labels, axis=0)

    remaining = total_samples - start_idx
    gen = load_generator_for_class(target_class)

    batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(batches):
        curr_batch = min(BATCH_SIZE, remaining - i * BATCH_SIZE)
        z = torch.randn(curr_batch, LATENT_DIM, device=DEVICE)
        labels = torch.full((curr_batch,), target_class, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            batch_x = gen(z, labels).cpu().numpy()
        batch_y = np.full(curr_batch, target_class, dtype=np.int16)
        synth_data.append(batch_x)
        synth_labels.append(batch_y)

        # Save checkpoint after each batch
        np.savez(checkpoint_path,
                 x=np.concatenate(synth_data, axis=0),
                 y=np.concatenate(synth_labels, axis=0))

        print(f"Generated {min(start_idx + (i+1)*BATCH_SIZE, total_samples)}/{total_samples} for class {target_class}")

    synth_data = np.concatenate(synth_data, axis=0)
    synth_labels = np.concatenate(synth_labels, axis=0)
    print(f"✅ Finished class {target_class}, total samples: {synth_data.shape[0]}")
    return synth_data, synth_labels

# -----------------------------
# Main
# -----------------------------
def main():
    # Load original dataset labels to check distribution
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    unique, counts = np.unique(labels, return_counts=True)
    print("Original label distribution:", dict(zip(unique, counts)))

    for cls, total in TOTAL_SAMPLES_PER_CLASS.items():
        if cls == 2:  # Optional: skip the largest class
            continue
        generate_synthetic_for_class(cls, total)

if __name__ == "__main__":
    main()
