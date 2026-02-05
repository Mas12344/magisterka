import torch
from autoencoder import vae_loss, calculate_metrics


def run_inference_with_metrics(model, loader, config, device="cuda", max_batches=5):
    model.eval()
    model.to(device)
    results = []

    mse_w = config.get('mse_weight', 1.0)
    l1_w = config.get('l1_weight', 0.1)
    fft_w = config.get('fft_weight', 1.0)
    kl_w = config.get('max_beta', 1.0)
    contrastive_w = config.get('contrastive_weight', 1.0)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)

            reconstructed, latent, mu, logvar = model(x)
            metrics = calculate_metrics(reconstructed, x)
            _, losses = vae_loss(reconstructed, x, mu, logvar, mse_w, l1_w, fft_w, kl_w, contrastive_w)

            results.append({
                'input': x[0].cpu(),
                'output': reconstructed[0].cpu(),
                'difference': torch.abs(x[0].cpu() - reconstructed[0].cpu()),
                'metrics': metrics,
                'losses': losses,
                'latent': latent[0].cpu(),
            })
    return results