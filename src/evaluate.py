import torch
from torch.utils.data import DataLoader
import os
from train import SimpleCNN  # import model architecture
from src.data_loader import ApneaSpectrogramDataset  # use real dataset

# --------------------------
# 1. Test Dataset
# --------------------------
def get_test_dataset(processed_folder="data/processed"):
    return ApneaSpectrogramDataset(processed_folder=processed_folder)

# --------------------------
# 2. Evaluation Function
# --------------------------
def evaluate_model(checkpoint_path="checkpoints/model_epoch5.pt", batch_size=16):
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Load model
    model = SimpleCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.eval()

    # Load test data
    test_dataset = get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"✅ Evaluation Complete — Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()