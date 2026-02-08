import torch
from autoencoder import vae_loss, calculate_metrics, augment_with_noise


def run_inference_with_metrics(model, loader, config, device="cuda", max_batches=5):
    model.eval()
    model.to(device)
    results = []

    mse_w = config.get('mse_weight', 1.0)
    l1_w = config.get('l1_weight', 0.1)
    fft_w = config.get('fft_weight', 1.0)
    kl_w = config.get('max_beta', 1.0)
    contrastive_w = config.get('contrastive_weight', 1.0)
    contrastive_temperature = config.get('contrastive_temperature', 0.5)
    augmentation_noise_std = config.get('augmentation_noise_std', 0.02)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)

            reconstructed, latent, mu, logvar, proj1 = model(x, return_projection=True)
            x_aug = augment_with_noise(x, noise_std=augmentation_noise_std)
            reconstructed_aug, _, _, _, proj2 = model(x_aug, return_projection=True)

            loss, parts = vae_loss(
                reconstructed, x, mu, logvar, 
                mse_w, l1_w, fft_w, 
                kl_w, 
                contrastive_w,
                proj1,
                proj2,
                contrastive_temperature
            )

            metrics = calculate_metrics(reconstructed, x)
            results.append({
                'input': x[0].cpu(),
                'input_aug': x_aug[0].cpu(),
                'output': reconstructed[0].cpu(),
                'output_aug': reconstructed_aug[0].cpu(),
                'difference': torch.abs(x[0].cpu() - reconstructed[0].cpu()),
                'metrics': metrics,
                'losses': parts,
                'latent': latent[0].cpu(),
            })
    return results