import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_loss(pred, target):
    pred_fft = torch.fft.rfft(pred, dim=3)
    target_fft = torch.fft.rfft(target, dim=3)

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    scale = torch.mode(torch.flatten(target_mag))[0]

    pred_mag /= scale
    target_mag /= scale

    # if pred_mag.isnan().any():
    #     print("nan w pred")
    #     print(scale)
    #     print(pred)
    #     print(pred_fft)
    #     print(pred_mag)
    #     exit()
    # if target_mag.isnan().any():
    #     print("nan w target")
    #     print(scale)
    #     print(target_mag)
    #     exit()

    return torch.mean((pred_mag - target_mag) ** 2)


def combined_loss(pred, target, mse_weight=1.0, l1_weight=1.0, fft_weight=1.0):
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
        'recon': total_loss.item(),
    }


def vae_loss(pred, target, mu, logvar, mse_weight=1.0, l1_weight=1.0, fft_weight=1.0, kl_weight=0.5, contrastive_weight=0):
    recon_loss, stats = combined_loss(pred, target, mse_weight, l1_weight, fft_weight)
    kl_loss = -.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    contrastive_loss = torch.tensor(1) # TODO

    total = recon_loss + kl_weight * kl_loss + contrastive_weight * contrastive_loss
    stats.update({'total': total.item(),'kl': kl_loss.item() * kl_weight, 'contrastive': contrastive_weight * contrastive_loss.item()})
    return total, stats


def calculate_metrics(pred, target):
    with torch.no_grad():
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        mae = F.l1_loss(pred, target)
    return {'psnr': psnr.item(), 'mae': mae.item()}


class StrainRateVAE(nn.Module):
    def __init__(self, input_size=512, latent_dim=2048, use_layernorm=True, dropout_rate=0.05):
        super(StrainRateVAE, self).__init__()

        self.latent_dim = latent_dim
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        self.input_size = input_size

        self.final_channels = 512
        self.final_spatial_size = self._calculate_final_size()

        self.encoder = self._build_encoder()
        self.flatten_size = self.final_channels * self.final_spatial_size * self.final_spatial_size
        self.fc_encode_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_encode_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = self._build_decoder()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _calculate_final_size(self):
        channels = [1, 32, 64, 128, 256, 512]
        spatial_size = self.input_size
        
        for i in range(len(channels) - 1):
            spatial_size = self._calculate_output_size(spatial_size)
        
        return spatial_size

    def _calculate_output_size(self, input_size, kernel_size=3, stride=2, padding=1):
        return (input_size + 2 * padding - kernel_size) // stride + 1

    def _conv_block(self, in_ch, out_ch, spatial_size, use_ln, dropout):
        layers = [nn.Conv2d(in_ch, out_ch, 3, 2, 1)]
        new_size = self._calculate_output_size(spatial_size)
        layers.append(nn.GELU())
        if use_ln:
            layers.append(nn.LayerNorm([out_ch, new_size, new_size]))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return layers, new_size

    def _build_encoder(self):
        layers = []
        channels = [1, 32, 64, 128, 256, 512]
        spatial_size = self.input_size
        for i in range(len(channels) - 1):
            block_layers, spatial_size = self._conv_block(
                channels[i], channels[i + 1],
                spatial_size,
                self.use_layernorm,
                self.dropout_rate
            )
            layers += block_layers
        return nn.Sequential(*layers)

    def _deconv_block(self, in_ch, out_ch):
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        ]
        layers.append(nn.GELU())
        return layers

    def _build_decoder(self):
        layers = []
        channels = [512, 256, 128, 64, 32, 1]
        for i in range(len(channels) - 1):
            layers += self._deconv_block(channels[i], channels[i + 1])
        return nn.Sequential(*layers)

    def reparam(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_encode_mu(x)
        logvar = self.fc_encode_logvar(x)
        return mu, logvar

    def decode(self, latent):
        x = self.fc_decode(latent)
        x = x.view(x.size(0), self.final_channels, self.final_spatial_size, self.final_spatial_size)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.reparam(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, latent, mu, logvar


if __name__ == '__main__':
    pred, target = torch.rand([1, 1, 512, 512]), torch.rand([1, 1, 512, 512])
    fft_loss(pred, target)
    srae = StrainRateVAE()
    channels = [1, 32, 64, 128, 256, 512]
    spatial_size = 128
    for i in range(len(channels) - 1):
        spatial_size = srae._calculate_output_size(spatial_size)
        print(spatial_size)

    reconstructed, latent, mu, logvar = srae(target)
    loss = vae_loss(reconstructed, target, mu, logvar)
    print(loss)

