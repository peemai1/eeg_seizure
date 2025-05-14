import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.dataset import SubjectDataset
from models.lstm import LSTMModel
from utils.eval import evaluate_model

# === CONFIG ===
data_folders = {
    'cleaned': 'data/processed/cleaned',
    'uncleaned': 'data/processed/uncleaned'
}

# Argument parsing for CLI use
parser = argparse.ArgumentParser()
parser.add_argument("--split_path", default="data/processed/split.json", help="Path to split.json")
args = parser.parse_args()
split_path = args.split_path


split_path = 'data/processed/split.json'  # Contains train/test file names (same for both versions)

# Hyperparameters
epochs = 1 #10
batch_size = 6 #8
learning_rate = 1e-3
seq_len = 640
input_size = 19
hidden_size = 128
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load split.json
with open(split_path, 'r') as f:
    split = json.load(f)
train_files = split['train']
test_files = split['test']

results = {}

# === TRAIN BOTH CLEANED AND UNCLEANED VERSIONS ===
for version, folder in data_folders.items():
    print(f"\n--- Training on {version.upper()} data ---")

    # Convert base filenames to full paths
    train_paths = [os.path.join(folder, f) for f in train_files]
    test_paths = [os.path.join(folder, f) for f in test_files]

    train_set = SubjectDataset(train_paths)
    val_set = SubjectDataset(test_paths)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=seq_len
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    metrics = evaluate_model(model, val_loader, device)
    print("📟 Validation Metrics:", metrics)

    results[version] = {
        'losses': losses,
        'metrics': metrics,
    }

    torch.save(model.state_dict(), f"models/lstm_{version}.pt")

# === PLOT LOSS CURVES ===
plt.figure(figsize=(8, 5))
for version, res in results.items():
    plt.plot(res['losses'], label=f"{version} loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_comparison.png")
plt.show()

# === PRINT FINAL METRICS ===
print("\n=== Final Evaluation Metrics ===")
for version, res in results.items():
    print(f"{version.upper()}: {res['metrics']}")
