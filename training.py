import os
import time
import torch
import torch.optim as optim
from torch import autograd
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch


from autoencoder import vae_loss, combined_loss, calculate_metrics

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


def flatten_grads(params):
    return torch.cat([
        p.grad.flatten()
        for p in params
        if p.grad is not None
    ])

def flatten_params(params):
    return torch.cat([
        p.data.flatten()
        for p in params
    ])



def compute_layer_grad_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[f"grad_norm/{name}"] = param.grad.norm().item()
            norms[f"weight_grad_ratio/{name}"] = (param.data.norm() / (param.grad.norm() + 1e-12)).item()
    return norms


def compute_gradient_similarity(current_grad, prev_grad):
    if prev_grad is None:
        return None
    
    cos_sim = torch.nn.functional.cosine_similarity(
        current_grad.unsqueeze(0), 
        prev_grad.unsqueeze(0)
    ).item()
    return cos_sim


class BetaScheduler:
    def __init__(self, cycle_length, type="linear", proportion=None):
        self.i = 0
        self.cycle_length = cycle_length
        self.proportion = max(min(proportion,1),0)
        if type == "linear":
            self.f = lambda : self.i/self.cycle_length
        elif type == "half":
            self.f = lambda : min(1,self.i/(self.cycle_length*self.proportion))

    def step(self):
        self.i += 1
        self.i %= (self.cycle_length+1)

    def get_beta(self):
        return self.f()


def train_autoencoder(model, train_loader, val_loader, config, device):
    num_epochs = config['num_epochs']
    beta_sch = BetaScheduler(num_epochs//2, "half", 0.5)
    
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
    con_w = config.get('contrastive_weight', 0.0)

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
        'latent_dim': config.get('latent_dim'),
        'learning_rate': lr,
        'min_lr': min_lr,
        'mse_weight': mse_w,
        'l1_weight': l1_w,
        'fft_weight': fft_w,
        'input_size': getattr(model, 'input_size', None),
        'dropout_rate': getattr(model, 'dropout_rate', None),
        'use_layernorm': getattr(model, 'use_layernorm', None),
        'device': str(device),
    })

    model.to(device)
    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_recon', 'train_mse', 'train_l1', 'train_fft', 'train_kl', 'train_contrastive',
        'val_recon', 'val_mse', 'val_l1', 'val_fft', 'val_kl', 'val_contrastive', 'train_psnr', 'train_mae',
        'val_psnr', 'val_mae', 'lr', 'grad_norm'
    ]}

    global_step = 0
    prev_grad = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'mse': 0, 'l1': 0, 'fft': 0, 'kl': 0, 'contrastive': 0}
        train_metrics = {'psnr': 0, 'mae': 0}
        epoch_lrs = []
        epoch_grad_norms = []
        epoch_grad_similarities = []

        for batch_idx, data in enumerate(train_loader):
            data = data[0] if isinstance(data, (list, tuple)) else data
            data = data.to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            optimizer.zero_grad()

            if mixed_precision:
                with autocast('cuda'):
                    out, _, mu, logvar, = model(data)
                    loss, parts = vae_loss(out, data, mu, logvar, mse_w, l1_w, fft_w, beta_sch.get_beta(), con_w)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out, _, mu, logvar, = model(data)
                loss, parts = vae_loss(out, data, mu, logvar, mse_w, l1_w, fft_w, beta_sch.get_beta(), con_w)
                loss.backward()
                optimizer.step()

            current_grad = flatten_grads(model.parameters())
            grad_norm = current_grad.norm().item()

            grad_similarity = compute_gradient_similarity(current_grad, prev_grad)
            if grad_similarity is not None:
                epoch_grad_similarities.append(grad_similarity)

            layer_norms = compute_layer_grad_norms(model)
            

            for k in train_losses:
                train_losses[k] += parts.get(k, loss.item())
            metrics = calculate_metrics(out, data)
            for k in train_metrics:
                train_metrics[k] += metrics[k]


            current_lr = optimizer.param_groups[0]['lr']
            epoch_lrs.append(current_lr)
            epoch_grad_norms.append(grad_norm)

            step_metrics = {
                "step/train_loss": parts.get("total", loss.item()),
                "step/train_recon": parts.get("recon"),
                "step/train_mse": parts.get("mse"),
                "step/train_l1": parts.get("l1"),
                "step/train_fft": parts.get("fft"),
                "step/train_kl": parts.get("kl"),
                "step/train_contrastive": parts.get("contrastive"),
                "step/train_psnr": metrics["psnr"],
                "step/train_mae": metrics["mae"],
                "step/lr": current_lr,
                "step/grad_norm": grad_norm,
            }

            if grad_similarity is not None:
                step_metrics["step/grad_similarity"] = grad_similarity

            step_metrics.update(layer_norms)
            mlflow.log_metrics(step_metrics, step=global_step)

            prev_grad = current_grad.detach().clone()
            global_step += 1


        scheduler.step()
        n = len(train_loader)
        avg_train = {k: v / n for k, v in train_losses.items()}
        avg_metrics = {k: v / n for k, v in train_metrics.items()}
        avg_lr = sum(epoch_lrs)/n
        avg_grad_norm = sum(epoch_grad_norms)/n

        history['train_loss'].append(avg_train['total'])
        history['train_recon'].append(avg_train['recon'])
        history['train_mse'].append(avg_train['mse'])
        history['train_l1'].append(avg_train['l1'])
        history['train_fft'].append(avg_train['fft'])
        history['train_kl'].append(avg_train['kl'])
        history['train_contrastive'].append(avg_train['contrastive'])
        history['train_psnr'].append(avg_metrics['psnr'])
        history['train_mae'].append(avg_metrics['mae'])
        history['lr'].append(avg_lr)
        history['grad_norm'].append(avg_grad_norm)


        val_results = validate(model, val_loader, mse_w, l1_w, fft_w, beta_sch.get_beta(), con_w, device, mixed_precision)
        
        beta_sch.step()

        for k, v in val_results.items():
            history[k].append(v)


        epoch_metrics = {
            "epoch/train_loss": avg_train["total"],
            "epoch/train_recon": avg_train["recon"],
            "epoch/train_mse": avg_train["mse"],
            "epoch/train_l1": avg_train["l1"],
            "epoch/train_fft": avg_train["fft"],
            "epoch/train_kl": avg_train["kl"],
            "epoch/train_contrastive": avg_train["contrastive"],
            "epoch/train_psnr": avg_metrics["psnr"],
            "epoch/train_mae": avg_metrics["mae"],
            "epoch/lr": avg_lr,
            "epoch/grad_norm": avg_grad_norm,
            "epoch/val_loss": val_results["val_loss"],
            "epoch/val_recon": val_results["val_recon"],
            "epoch/val_mse": val_results["val_mse"],
            "epoch/val_l1": val_results["val_l1"],
            "epoch/val_fft": val_results["val_fft"],
            "epoch/val_kl": val_results["val_kl"],
            "epoch/val_contrastive": val_results["val_contrastive"],
            "epoch/val_psnr": val_results["val_psnr"],
            "epoch/val_mae": val_results["val_mae"],
        }

        if epoch_grad_similarities:
            epoch_metrics["epoch/grad_similarity"] = sum(epoch_grad_similarities)/n
        
        mlflow.log_metrics(epoch_metrics, step=epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train: {avg_train['total']:.6f}, Val: {val_results['val_loss']:.6f}, "
              f"LR: {avg_lr:.6e}, GradNorm: {avg_grad_norm:.4f}")

        if epoch_grad_similarities:
            print(f"  Grad Similarity: {sum(epoch_grad_similarities)/n:.4f}")

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

    print("training completed")
    return history


def validate(model, loader, mse_w, l1_w, fft_w, kl_w, con_w, device, mixed_precision):
    model.eval()
    losses = {'total': 0, 'recon': 0, 'mse': 0, 'l1': 0, 'fft': 0, 'kl': 0, 'contrastive': 0}
    metrics = {'psnr': 0, 'mae': 0}

    with torch.no_grad():
        for data in loader:
            data = data[0] if isinstance(data, (list, tuple)) else data
            data = data.to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            if mixed_precision:
                with autocast('cuda'):
                    out, _, mu, logvar, = model(data)
                    loss, parts = vae_loss(out, data, mu, logvar, mse_w, l1_w, fft_w, kl_w, con_w)
            else:
                out, _, mu, logvar, = model(data)
                loss, parts = vae_loss(out, data, mu, logvar, mse_w, l1_w, fft_w, kl_w, con_w)

            for k in losses:
                losses[k] += parts.get(k, loss.item())
            vals = calculate_metrics(out, data)
            for k in metrics:
                metrics[k] += vals[k]

    n = len(loader)
    return {
        'val_loss': losses['total'] / n,
        'val_recon': losses['recon'] / n,
        'val_mse': losses['mse'] / n,
        'val_l1': losses['l1'] / n,
        'val_fft': losses['fft'] / n,
        'val_kl': losses['kl'] / n,
        'val_contrastive': losses['contrastive'] / n,
        'val_psnr': metrics['psnr'] / n,
        'val_mae': metrics['mae'] / n,
    }


if __name__ == '__main__':
    
    b = BetaScheduler(97, "half", 0.5)
    for _ in range(20):
        print(b.get_beta())
        [b.step() for _ in range(10)]
