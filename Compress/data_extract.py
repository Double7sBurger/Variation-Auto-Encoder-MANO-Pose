import pickle
import numpy as np
from pathlib import Path
# This extract data from the s1-s10 whole dataset and concatenate all of them.
all_theta = []

for pkl in Path("../hand_dataset").glob("s*.pkl"):
    with open(pkl, "rb") as f:
        data = pickle.load(f)      # list of (Ni, 45)
        theta = np.concatenate(data, axis=0)
        all_theta.append(theta)

theta = np.concatenate(all_theta, axis=0)


theta = theta.astype(np.float32)
# theta.shape:(1623695, 45)
mean = theta.mean(axis=0)
std  = theta.std(axis=0) + 1e-6

theta_norm = (theta - mean) / std

np.savez(
    "../grab_mano_theta_all_norm.npz",
    theta=theta_norm,
    mean=mean,
    std=std
)
