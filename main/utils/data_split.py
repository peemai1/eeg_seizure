import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def extract_label(npz_path):
    """
    Load .npz file and derive a binary label from y.
    Assumes y is shape (N, 640) with binary values (0 or 1).
    Uses majority voting across all frames.
    """
    data = np.load(npz_path)
    y = data['y']  # shape (N, 640)
    # Flatten and average across all labels
    majority_label = int(np.round(y.mean()))
    return majority_label

def stratified_split(base_dir, test_ratio=0.2, seed=42, save_path="split.json"):
    """
    Performs stratified split of .npz files based on labels.
    """
    all_files = [f for f in os.listdir(base_dir) if f.endswith('.npz')]
    all_paths = [os.path.join(base_dir, f) for f in all_files]

    labels = []
    for path in all_paths:
        try:
            label = extract_label(path)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if len(labels) != len(all_paths):
        raise ValueError("Mismatch in number of files and extracted labels.")

    train_paths, test_paths = train_test_split(
        all_files,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )

    split = {
        "train": sorted(train_paths),
        "test": sorted(test_paths)
    }

    with open(save_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"✅ Stratified split saved to {save_path}: {len(train_paths)} train, {len(test_paths)} test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified split of .npz files by label.")
    parser.add_argument("base_dir", help="Directory containing .npz files.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_path", default="split.json", help="Where to save the split JSON.")
    args = parser.parse_args()

    stratified_split(args.base_dir, args.test_ratio, args.seed, args.save_path)