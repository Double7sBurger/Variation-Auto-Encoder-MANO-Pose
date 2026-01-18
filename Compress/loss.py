import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, beta=1e-3):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = recon_loss + beta * kl
    return total_loss, recon_loss, kl
