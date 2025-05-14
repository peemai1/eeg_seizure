import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # <-- added tqdm import

class SubjectDataset(Dataset):
    def __init__(self, file_paths, verbose=True):
        """
        file_paths: List of full paths to .npz files
        """
        self.X = []
        self.y = []
        self.subject_ids = []

        if verbose:
            print(f"Loading {len(file_paths)} .npz files...")
            iterator = tqdm(file_paths, desc="Loading .npz files")
        else:
            iterator = file_paths

        for path in iterator:
            try:
                data = np.load(path)
                self.X.append(data['X'])  # shape: (num_windows, 19, window_size)
                self.y.append(data['y'])  # shape: (num_windows, window_size)

                subject_id = data.get('subject_id', os.path.basename(path).split('.')[0])
                self.subject_ids.extend([subject_id] * data['X'].shape[0])

                if verbose:
                    tqdm.write(f"Loaded: {path} → {data['X'].shape[0]} samples")

            except Exception as e:
                tqdm.write(f"Failed to load {path}: {e}")

        if not self.X:
            raise RuntimeError("No valid .npz files loaded.")

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
