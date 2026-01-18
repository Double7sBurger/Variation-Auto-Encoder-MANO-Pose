import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from vae import HandPoseVAE

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    data_npz = "theta/grab_mano_theta_all_norm.npz"   # Training set
    ckpt     = "hand_pose_vae_all_z9.pt"              # Trained weights
    outdir   = "outputs/recon_vis_z9"
    os.makedirs(outdir, exist_ok=True)

    # 1) load normalized theta (N,45)
    data = np.load(data_npz)
    theta = data["theta"].astype(np.float32)  # normalized already
    N = theta.shape[0]
    print("theta:", theta.shape, "min/max:", theta.min(), theta.max())

    # 2) load model
    latent_dim = 9
    model = HandPoseVAE(input_dim=45, latent_dim=latent_dim).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # 3) sample B examples
    B = 16
    idx = np.random.choice(N, size=B, replace=False)
    x = torch.from_numpy(theta[idx]).to(device)

    # recon (normalized space)
    recon, mu, logvar = model(x)
    x_np = x.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # 4) metrics
    mse = np.mean((recon_np - x_np) ** 2)
    mae = np.mean(np.abs(recon_np - x_np))
    print(f"Recon on normalized theta: MSE={mse:.6f}, MAE={mae:.6f}")

    # 5) overlay plots per sample
    for i in range(B):
        per_mse = np.mean((recon_np[i] - x_np[i]) ** 2)
        plt.figure(figsize=(10, 3))
        plt.plot(x_np[i], label="GT (theta_norm)")
        plt.plot(recon_np[i], label="Recon (theta_norm_hat)")
        plt.legend()
        plt.title(f"Sample {i}  MSE={per_mse:.6f}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"theta_overlay_{i:02d}.png"))
        plt.close()

    # 6) per-dimension error bar
    dim_mse = np.mean((recon_np - x_np) ** 2, axis=0)
    plt.figure(figsize=(12, 3))
    plt.bar(np.arange(45), dim_mse)
    plt.title("Per-dimension MSE (normalized space)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_dim_mse.png"))
    plt.close()

    # 7) latent distribution quick check
    mu_np = mu.cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.boxplot([mu_np[:, j] for j in range(mu_np.shape[1])], showfliers=False)
    plt.title("Latent mu distribution (batch)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latent_mu_boxplot.png"))
    plt.close()

    print("Saved to:", outdir)

if __name__ == "__main__":
    main()
