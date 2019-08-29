import torch
from torch import nn

from flatten_layer import Flatten
from reshape_layer import Reshape


class Module_dSprites_VAE(nn.Module):
    def __init__(self, channels=1, latent_dim=10, filters_first_layer=32):
        self.channels = channels
        self.filters_first_layer = filters_first_layer
        self.latent_dim = latent_dim
        super().__init__()
        self.encoder = self.define_frames_encoder(latent_dim=self.latent_dim)
        self.decoder = self.define_frames_decoder(latent_dim=self.latent_dim)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.decoder[-2].weight, nn.init.calculate_gain('sigmoid'))

    def conv_unit(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=False,
                  activation_relu=True):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation_relu:
            modules.append(nn.ReLU(True))
        return nn.Sequential(*modules)

    def define_frames_encoder(self, latent_dim=200):
        modules = []
        filters = self.filters_first_layer
        in_ch = self.channels
        out_ch = filters
        for _ in range(4):
            modules.append(self.conv_unit(in_ch, out_ch, 4, 2, 1, batch_norm=True, activation_relu=True))
            in_ch = out_ch
            out_ch *= 2
        modules.append(self.conv_unit(in_ch, in_ch, 4, 1, batch_norm=True, activation_relu=True))
        modules.append(self.conv_unit(in_ch, latent_dim * 2, 1, batch_norm=False, activation_relu=False))
        # modules.append(Flatten())
        # modules = modules + [
        #     nn.Linear(4 * 4 * self.filters_first_layer * 8, latent_dim * 2),
        #     nn.BatchNorm1d(latent_dim * 2),
        #     nn.ReLU(True)
        # ]
        return nn.Sequential(*modules)

    def deconv_unit(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=False,
                    activation_relu=True):
        modules = list()
        modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation_relu:
            modules.append(nn.ReLU(True))
        return nn.Sequential(*modules)

    def define_frames_decoder(self, latent_dim=512):
        # modules = [
        #     nn.Linear(latent_dim * 2, 4 * 4 * self.filters_first_layer * 8),
        #     nn.BatchNorm1d(4 * 4 * self.filters_first_layer * 8),
        #     nn.ReLU(True),
        #     Reshape((), (self.filters_first_layer * 8, 4, 4))
        # ]
        in_ch = self.filters_first_layer * 8
        out_ch = int(in_ch) // 2
        modules = [
            self.deconv_unit(latent_dim, in_ch, 1, batch_norm=True, activation_relu=True),
            self.deconv_unit(in_ch, in_ch, 4, batch_norm=True, activation_relu=True)
        ]
        for _ in range(3):
            modules.append(self.deconv_unit(in_ch, out_ch, 4, 2, 1, batch_norm=True, activation_relu=True))
            in_ch = out_ch
            out_ch = int(in_ch) // 2
        modules = modules + [
            nn.ConvTranspose2d(self.filters_first_layer, self.channels, 4, 2, 1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        std = (0.5 * logvar).exp()
        return mu + eps * std

    def forward(self, x, no_dec=False):
        x_encoded = self.encoder(x)
        z_mean = x_encoded[:, :self.latent_dim]
        z_logvar = x_encoded[:, self.latent_dim:]
        z = self.reparameterize(z_mean, z_logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decoder(z)
            return x_recon, z_mean, z_logvar, z.squeeze()
