import torch
from torch import nn


class Module_dSprites_VAE(nn.Module):
    def __init__(self, channels=1, latent_dim=10, filters_first_layer=32):
        self.channels = channels
        self.filters_first_layer = filters_first_layer
        self.latent_dim = latent_dim
        super().__init__()
        self.encode = self.define_frames_encoder(latent_dim=self.latent_dim)
        self.decode = self.define_frames_decoder(latent_dim=self.latent_dim)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # nn.init.xavier_normal_(self.decode[-2].weight, nn.init.calculate_gain('sigmoid'))

    def define_frames_encoder(self, latent_dim=200):
        net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Conv2d(128, 2 * latent_dim, kernel_size=1)
        )
        return net

    def define_frames_decoder(self, latent_dim=512):
        net = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        return net

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        std = (0.5 * logvar).exp()
        return mu + eps * std

    def forward(self, x, no_dec=False):
        x_encoded = self.encode(x)
        z_mean = x_encoded[:, :self.latent_dim]
        z_logvar = x_encoded[:, self.latent_dim:]
        z = self.reparameterize(z_mean, z_logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, z_mean, z_logvar, z.squeeze()
