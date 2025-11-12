import torch
from autoencoder import combined_loss, calculate_metrics


def run_inference_with_metrics(model, loader, device="cuda", max_batches=5):
    model.eval()
    model.to(device)
    results = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)

            reconstructed, latent = model(x)
            metrics = calculate_metrics(reconstructed, x)
            _, losses = combined_loss(reconstructed, x)

            results.append({
                'input': x[0].cpu(),
                'output': reconstructed[0].cpu(),
                'difference': torch.abs(x[0].cpu() - reconstructed[0].cpu()),
                'metrics': metrics,
                'losses': losses,
                'latent': latent[0].cpu(),
            })
    return results
