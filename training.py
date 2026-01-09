import os
import time
import torch
from torch.optim import Optimizer
from torch import autograd
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


class AdaptiveLR(Optimizer):
    def __init__(
        self,
        params,
        target_ratio=1e-3,
        damping=1e-6,
        ema_beta=0.9,
        min_lr=1e-6,
        max_lr=1.0,
        use_curvature=True,
        lr_change_limit=2.0,
        curvature_scale=0.3
    ):
        super().__init__(params, defaults={})

        self.target_ratio = target_ratio
        self.damping = damping
        self.ema_beta = ema_beta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.use_curvature = use_curvature
        self.curvature_scale = curvature_scale

        self.state["lr_ema"] = None
        self.state["last_lr"] = None
        self.state["last_gHg"] = None

    def _lr_from_relative_step(self, params, eps=1e-12):
        g = flatten_grads(params)
        w = flatten_params(params)
        return self.target_ratio * (w.norm() / (g.norm() + eps))

    def _lr_from_gHg(self, loss, params):
        params = [p for p in params if p.requires_grad]

        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True
        )

        g = torch.cat([gi.flatten() for gi in grads])
        assert g.requires_grad

        Hv = torch.autograd.grad(
            outputs=g,
            inputs=params,
            grad_outputs=g,
            retain_graph=True
        )

        Hv_flat = torch.cat([h.flatten() for h in Hv])
        gHg = torch.dot(g, Hv_flat)

        lr = g.norm()**2 / (gHg + self.damping)
        return lr, gHg


    def _ema_lr(self, lr):
        if self.state["lr_ema"] is None:
            self.state["lr_ema"] = lr.detach()
        else:
            self.state["lr_ema"] = (
                self.ema_beta * self.state["lr_ema"]
                + (1 - self.ema_beta) * lr
            ).detach()
        return self.state["lr_ema"]


    def step(self, closure):
        with torch.enable_grad():
            loss, rest = closure()

        params = self.param_groups[0]["params"]
        params = [p for p in params if p.requires_grad]

        lr_rel = self._lr_from_relative_step(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)

        lr = lr_rel
        gHg = None

        if self.use_curvature:
            lr_curv, gHg_val = self._lr_from_gHg(loss, params)
            
            if gHg_val is not None and gHg_val > 0 and torch.isfinite(lr_curv):
                lr = torch.minimum(lr_rel, self.curvature_scale * lr_curv)
                gHg = gHg_val
            
            self.state["last_gHg"] = gHg.item() if gHg is not None else None
        else:
            lr = lr_rel
            self.state["last_gHg"] = None


        prev_lr = self.state["last_lr"]
        if prev_lr is not None:
            lr = torch.minimum(lr, prev_lr)
        
        if prev_lr is not None:
            lr = self._ema_lr(lr)
        else:
            self._ema_lr(lr)

        if prev_lr is not None:
            lower = prev_lr / 2
            upper = prev_lr ** 2
            lr = torch.clamp(lr, lower, upper)

        lr = torch.clamp(lr, self.min_lr, self.max_lr)
        
        self.state["last_lr"] = lr.detach()
        return loss, rest



def train_autoencoder(model, train_loader, val_loader, config, device):
    num_epochs = config['num_epochs']
    
    use_adaptive_lr = config.get('use_adaptive_lr', True)
    adaptive_method = config.get('adaptive_method', 'relative')
    target_ratio = config.get('target_ratio', 1e-3)
    curvature_damping = config.get('curvature_damping', 1e-6)
    lr_ema_beta = config.get('lr_ema_beta', 0.9)
    min_lr = config.get('min_lr', 1e-9)
    max_lr = config.get('max_lr', 2)

    adam = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,          # placeholder
        weight_decay=config['weight_decay']
    )

    adaptive_lr = AdaptiveLR(
        model.parameters(),
        target_ratio=target_ratio,
        damping=curvature_damping,
        ema_beta=lr_ema_beta,
        min_lr=min_lr,
        max_lr=max_lr,
        use_curvature=(adaptive_method != 'relative')
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
        'use_adaptive_lr': use_adaptive_lr,
        'adaptive_method': adaptive_method,
        'target_ratio': target_ratio,
        'lr_ema_beta': lr_ema_beta,
        'latent_dim': config.get('latent_dim'),
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
        'train_loss', 'val_loss', 'train_mse', 'train_l1', 'train_fft',
        'val_mse', 'val_l1', 'val_fft', 'train_psnr', 'train_mae',
        'val_psnr', 'val_mae', 'lr', 'grad_norm', 'gHg_norm', 'lr_ema'
    ]}

    global_step = 0
    prev_grad = None

    print(f"training with adaptive LR (method: {adaptive_method})")


    for epoch in range(num_epochs):
        model.train()
        train_losses = {'total': 0, 'mse': 0, 'l1': 0, 'fft': 0}
        train_metrics = {'psnr': 0, 'mae': 0}
        epoch_lrs = []
        epoch_ema = []
        epoch_grad_norms = []
        epoch_gHg_norms = []
        epoch_grad_similarities = []

        for batch_idx, data in enumerate(train_loader):
            data = data[0] if isinstance(data, (list, tuple)) else data
            data = data.to(device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            adam.zero_grad()

            def closure():
                out, _ = model(data)
                loss, parts = combined_loss(out, data, mse_w, l1_w, fft_w)
                loss.backward(
                    retain_graph=adaptive_method in ['curvature', 'hybrid'],
                )
                assert loss.requires_grad
                return loss, (parts, out)

            loss, (parts, out) = adaptive_lr.step(closure)

            lr = adaptive_lr.state["last_lr"]
            for g in adam.param_groups:
                g["lr"] = lr


            if mixed_precision:
                scaler.update()

            current_lr = adaptive_lr.state["last_lr"]
            gHg_value = adaptive_lr.state["last_gHg"]
            ema_value = adaptive_lr.state["lr_ema"]

            current_grad = flatten_grads(model.parameters())
            grad_norm = current_grad.norm().item()

            grad_similarity = compute_gradient_similarity(current_grad, prev_grad)
            if grad_similarity is not None:
                epoch_grad_similarities.append(grad_similarity)

            layer_norms = compute_layer_grad_norms(model)
            
            adam.step()
            #adam.zero_grad()

            for k in train_losses:
                train_losses[k] += parts.get(k, loss.item())
            metrics = calculate_metrics(out, data)
            for k in train_metrics:
                train_metrics[k] += metrics[k]

            epoch_lrs.append(current_lr)
            epoch_ema.append(ema_value)
            epoch_grad_norms.append(grad_norm)
            if gHg_value is not None:
                epoch_gHg_norms.append(abs(gHg_value))

            step_metrics = {
                "step/train_loss": parts.get("total", loss.item()),
                "step/train_mse": parts.get("mse"),
                "step/train_l1": parts.get("l1"),
                "step/train_fft": parts.get("fft"),
                "step/train_psnr": metrics["psnr"],
                "step/train_mae": metrics["mae"],
                "step/lr": current_lr,
                "step/grad_norm": grad_norm,
                "step/lr_ema": ema_value
            }

            if grad_similarity is not None:
                step_metrics["step/grad_similarity"] = grad_similarity
            if gHg_value is not None:
                step_metrics["step/gHg"] = abs(gHg_value)

            step_metrics.update(layer_norms)
            mlflow.log_metrics(step_metrics, step=global_step)

            prev_grad = current_grad.detach().clone()
            global_step += 1


        n = len(train_loader)
        avg_train = {k: v / n for k, v in train_losses.items()}
        avg_metrics = {k: v / n for k, v in train_metrics.items()}
        avg_lr = sum(epoch_lrs)/n
        avg_ema = sum(epoch_ema)/n
        avg_grad_norm = sum(epoch_grad_norms)/n

        history['train_loss'].append(avg_train['total'])
        history['train_mse'].append(avg_train['mse'])
        history['train_l1'].append(avg_train['l1'])
        history['train_fft'].append(avg_train['fft'])
        history['train_psnr'].append(avg_metrics['psnr'])
        history['train_mae'].append(avg_metrics['mae'])
        history['lr'].append(avg_lr.item())
        history['grad_norm'].append(avg_grad_norm)
        history['lr_ema'].append(avg_ema.item())

        if epoch_gHg_norms:
            avg_gHg = sum(epoch_gHg_norms)/n
            history['gHg_norm'].append(avg_gHg)

        val_results = validate(model, val_loader, mse_w, l1_w, fft_w, device, mixed_precision)
        for k, v in val_results.items():
            history[k].append(v)


        epoch_metrics = {
            "epoch/train_loss": avg_train["total"],
            "epoch/train_mse": avg_train["mse"],
            "epoch/train_l1": avg_train["l1"],
            "epoch/train_fft": avg_train["fft"],
            "epoch/train_psnr": avg_metrics["psnr"],
            "epoch/train_mae": avg_metrics["mae"],
            "epoch/lr": avg_lr,
            "epoch/lr_ema": avg_ema,
            "epoch/grad_norm": avg_grad_norm,
            "epoch/val_loss": val_results["val_loss"],
            "epoch/val_mse": val_results["val_mse"],
            "epoch/val_l1": val_results["val_l1"],
            "epoch/val_fft": val_results["val_fft"],
            "epoch/val_psnr": val_results["val_psnr"],
            "epoch/val_mae": val_results["val_mae"],
        }

        if epoch_grad_similarities:
            epoch_metrics["epoch/grad_similarity"] = sum(epoch_grad_similarities)/n
        if epoch_gHg_norms:
            epoch_metrics["epoch/gHg"] = avg_gHg
        
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
