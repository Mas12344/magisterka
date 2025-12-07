import os
import time
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

from autoencoder import combined_loss, calculate_metrics

from torch_lr_finder import LRFinder

import torch.nn as nn
class LRFinderLoss(nn.Module):
    def __init__(self, mse, l1, fft):
        super().__init__()
        self.mse = mse
        self.l1 = l1
        self.fft = fft

    def forward(self, pred, target):
        pred, _ = pred
        return combined_loss(pred, target, mse_weight=self.mse, l1_weight=self.l1, fft_weight=self.fft)[0]


def find_lr(model, trainloader, device):
    mse_w = 1.0
    l1_w  = 1.0
    fft_w = 1.0
    criterion = LRFinderLoss(mse_w,l1_w,fft_w)
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.01)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=100, num_iter=1000)
    lr_finder.plot()
    lr_finder.reset()


def train_autoencoder(model, train_loader, val_loader, config, device):
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    min_lr = config.get('min_lr', 1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_lr
    )
    mixed_precision = config.get('mixed_precision', False)
    scaler = GradScaler('cuda') if mixed_precision else None

    mse_w = config.get('mse_weight', 1.0)
    l1_w = config.get('l1_weight', 0.1)
    fft_w = config.get('fft_weight', 1.0)

    save_every = config.get('save_every', 50)
    project_name = config.get('project_name', 'autoencoder')
    run_name = config.get('run_name', f'run_{int(time.time())}')
    use_tb = config.get('use_tensorboard', False)

    checkpoint_dir = f'checkpoints/{project_name}/{run_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = None
    if use_tb:
        log_dir = f'runs/{project_name}/{run_name}'
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    mlflow.log_params({
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'min_lr': min_lr,
        'latent_dim': config.get('latent_dim'),
        'mse_weight': mse_w,
        'l1_weight': l1_w,
        'fft_weight': fft_w,
        'dropout_rate': model.dropout_rate,
        'use_batchnorm': model.use_batchnorm,
        'device': str(device),
    })

    model.to(device)
    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_mse', 'train_l1', 'train_fft',
        'val_mse', 'val_l1', 'val_fft', 'train_psnr', 'train_mae',
        'val_psnr', 'val_mae', 'lr'
    ]}



    for epoch in range(num_epochs):
        model.train()
        train_losses = {'total': 0, 'mse': 0, 'l1': 0, 'fft': 0}
        train_metrics = {'psnr': 0, 'mae': 0}

        for data in train_loader:
            data = data[0] if isinstance(data, (list, tuple)) else data
            data = data.to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            optimizer.zero_grad()
            if mixed_precision:
                with autocast('cuda'):
                    out, _ = model(data)
                    loss, parts = combined_loss(out, data, mse_w, l1_w, fft_w)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out, _ = model(data)
                loss, parts = combined_loss(out, data, mse_w, l1_w, fft_w)
                loss.backward()
                optimizer.step()

            for k in train_losses:
                train_losses[k] += parts.get(k, loss.item())
            metrics = calculate_metrics(out, data)
            for k in train_metrics:
                train_metrics[k] += metrics[k]

        scheduler.step()
        n = len(train_loader)
        avg_train = {k: v / n for k, v in train_losses.items()}
        avg_metrics = {k: v / n for k, v in train_metrics.items()}

        history['train_loss'].append(avg_train['total'])
        history['train_mse'].append(avg_train['mse'])
        history['train_l1'].append(avg_train['l1'])
        history['train_fft'].append(avg_train['fft'])
        history['train_psnr'].append(avg_metrics['psnr'])
        history['train_mae'].append(avg_metrics['mae'])
        history['lr'].append(scheduler.get_last_lr()[0])

        val_results = validate(model, val_loader, mse_w, l1_w, fft_w, device, mixed_precision)
        for k, v in val_results.items():
            history[k].append(v)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train['total']:.6f}, Val Loss: {val_results['val_loss']:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6e}")

        if (epoch + 1) % save_every == 0:
            path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({'model_state_dict': model.state_dict()}, path)
            mlflow.log_artifact(path, artifact_path="checkpoints")

    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'history': history}, final_path)
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact(final_path, artifact_path="final_model")

    if writer:
        writer.close()
    return history


def validate(model, loader, mse_w, l1_w, fft_w, device, mixed_precision):
    model.eval()
    losses = {'total': 0, 'mse': 0, 'l1': 0, 'fft': 0}
    metrics = {'psnr': 0, 'mae': 0}

    with torch.no_grad():
        for data in loader:
            data = data[0] if isinstance(data, (list, tuple)) else data
            data = data.to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            if mixed_precision:
                with autocast('cuda'):
                    out, _ = model(data)
                    loss, parts = combined_loss(out, data, mse_w, l1_w, fft_w)
            else:
                out, _ = model(data)
                loss, parts = combined_loss(out, data, mse_w, l1_w, fft_w)

            for k in losses:
                losses[k] += parts.get(k, loss.item())
            vals = calculate_metrics(out, data)
            for k in metrics:
                metrics[k] += vals[k]

    n = len(loader)
    return {
        'val_loss': losses['total'] / n,
        'val_mse': losses['mse'] / n,
        'val_l1': losses['l1'] / n,
        'val_fft': losses['fft'] / n,
        'val_psnr': metrics['psnr'] / n,
        'val_mae': metrics['mae'] / n,
    }
