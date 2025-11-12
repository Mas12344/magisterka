import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    return torch.mean((pred_mag - target_mag) ** 2)


def combined_loss(pred, target, mse_weight=1.0, l1_weight=0.1, fft_weight=0.1):
    mse_loss = F.mse_loss(pred, target)
    l1_loss = F.l1_loss(pred, target)
    fft_component = fft_loss(pred, target)
    total_loss = (
        mse_weight * mse_loss
        + l1_weight * l1_loss
        + fft_weight * fft_component
    )
    return total_loss, {
        'mse': mse_loss.item() * mse_weight,
        'l1': l1_loss.item() * l1_weight,
        'fft': fft_component.item() * fft_weight,
        'total': total_loss.item(),
    }


def calculate_metrics(pred, target):
    with torch.no_grad():
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        mae = F.l1_loss(pred, target)
    return {'psnr': psnr.item(), 'mae': mae.item()}


class StrainRateAutoencoder(nn.Module):
    def __init__(self, latent_dim=2048, use_batchnorm=True, dropout_rate=0.05):
        super(StrainRateAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        self.encoder = self._build_encoder()
        self.flatten_size = 512 * 16 * 16
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = self._build_decoder()

    def _conv_block(self, in_ch, out_ch, use_bn, dropout):
        layers = [nn.Conv2d(in_ch, out_ch, 3, 2, 1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return layers

    def _build_encoder(self):
        layers = []
        channels = [1, 32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers += self._conv_block(channels[i], channels[i + 1],
                                       self.use_batchnorm, self.dropout_rate)
        return nn.Sequential(*layers)

    def _deconv_block(self, in_ch, out_ch, use_bn, dropout):
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return layers

    def _build_decoder(self):
        layers = []
        channels = [512, 256, 128, 64, 32, 1]
        for i in range(len(channels) - 1):
            layers += self._deconv_block(channels[i], channels[i + 1],
                                         self.use_batchnorm, self.dropout_rate)
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)

    def decode(self, latent):
        x = self.fc_decode(latent)
        x = x.view(x.size(0), 512, 16, 16)
        return self.decoder(x)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
