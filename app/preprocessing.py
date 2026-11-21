"""Training-only statistics and the input transform shared with the API."""
import numpy as np


def training_statistics(x):
    # Accumulate in float64, then save the float32 values actually used by torch.
    mu = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    sd = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    # A constant feature becomes zero after centring; dividing by one is safe.
    sd = np.where(sd > 0, sd, np.float32(1)).astype(np.float32)
    return mu, sd


def normalise_inputs(x, mu, sd):
    return (np.asarray(x, dtype=np.float32) - mu) / sd
