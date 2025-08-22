import torch
import numpy as np
from sklearn.metrics import roc_curve


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj  # non-tensor leaf
    
def fpr_at_95_tpr(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find the threshold where TPR >= 0.95
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return None  # No threshold achieves 95% TPR
    idx = idxs[0]  # first index where TPR >= 0.95
    return fpr[idx]


# Normalise
def preprocess(seqs, mu, std):
    seqs = [(s - mu) / std for s in seqs]
    return torch.stack(seqs)