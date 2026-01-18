import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from vae import HandPoseVAE
from loss import vae_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load data ----------
data = np.load("theta/grab_mano_theta_all_norm.npz")
theta = data["theta"]    # (N, 45)

theta = torch.from_numpy(theta).float()
dataset = TensorDataset(theta)

loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True,
    drop_last=True
)

# ---------- Model ----------
model = HandPoseVAE(input_dim=45, latent_dim=9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- Training ----------
epochs = 100
beta = 1e-3

for epoch in range(epochs):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)

        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"[{epoch:03d}] "
        f"loss={total_loss/len(loader):.6f} "
        f"recon={recon_l.item():.6f} "
        f"kl={kl_l.item():.6f}"
    )

torch.save(model.state_dict(), "hand_pose_vae_all_z9.pt")
