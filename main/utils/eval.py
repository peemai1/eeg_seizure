import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)  # shape: (B, 19, T)
            targets = targets.to(device)  # shape: (B, T)

            outputs = model(inputs)  # shape: (B, T)
            probs = torch.sigmoid(outputs)  # convert logits to probabilities
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0).flatten()
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_prob = np.concatenate([torch.sigmoid(model(x.to(device))).detach().cpu().numpy().flatten()
                             for x, _ in dataloader], axis=0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob)
    }

    return metrics
