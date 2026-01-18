import torch
import torch.nn as nn
import torch.nn.functional as F

class HandPoseVAE(nn.Module):
    def __init__(self, input_dim=45, latent_dim=12):
        super().__init__()

        # ---------- Encoder ----------
        self.enc_fc1 = nn.Linear(input_dim, 256)
        self.enc_fc2 = nn.Linear(256, 128)
        self.enc_mu  = nn.Linear(128, latent_dim)
        self.enc_logvar = nn.Linear(128, latent_dim)

        # ---------- Decoder ----------
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_out = nn.Linear(256, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    ## loss is defined in the loss.py file
