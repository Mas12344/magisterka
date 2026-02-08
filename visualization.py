import matplotlib.pyplot as plt
import mlflow
import numpy as np


def visualize_results(results, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(n, 6, figsize=(20, 4 * n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, res in enumerate(results):
        inp = res['input'].squeeze()
        inp_aug = res['input_aug'].squeeze()
        out = res['output'].squeeze()
        out_aug = res['output_aug'].squeeze()
        diff = res['difference'].squeeze()
        # diff_aug = torch.abs(inp_aug - out_aug)
        
        axes[i, 0].imshow(inp, cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(inp_aug, cmap='viridis')
        axes[i, 1].set_title(f'Input Aug {i+1}', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(out, cmap='viridis')
        axes[i, 2].set_title(f'Reconstructed {i+1}', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(out_aug, cmap='viridis')
        axes[i, 3].set_title(f'Reconstructed Aug {i+1}', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        
        im1 = axes[i, 4].imshow(diff, cmap='hot', vmin=0)
        axes[i, 4].set_title(f'Diff {i+1}', fontsize=12, fontweight='bold')
        axes[i, 4].axis('off')
        plt.colorbar(im1, ax=axes[i, 4], fraction=0.046, pad=0.04)
        
        m = res['metrics']
        l = res['losses']
        
        metrics_text = (
            f"Metrics:\n"
            f"  PSNR: {m['psnr']:.2f} dB\n"
            f"  MAE: {m['mae']:.6f}\n\n"
            f"Losses:\n"
            f"  MSE: {l['mse']:.6f}\n"
            f"  L1: {l['l1']:.6f}\n"
            f"  FFT: {l['fft']:.6f}\n"
            f"  KL: {l['kl']:.6f}\n"
            f"  Contrastive: {l['contrastive']:.6f}\n"
            f"  Total: {l['total']:.6f}"
        )
        
        axes[i, 5].text(
            0.05, 0.5, metrics_text,
            transform=axes[i, 5].transAxes,
            fontsize=10,
            va='center',
            ha='left',
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='#f0f0f0',
                edgecolor='#cccccc',
                alpha=0.9
            )
        )
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(save_path, artifact_path="visualizations")
    
    plt.close()


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history['train_mse'], label='Train MSE')
    axes[0, 1].plot(epochs, history['val_mse'], label='Val MSE')
    axes[0, 1].set_title('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, history['train_psnr'], label='Train PSNR')
    axes[1, 0].plot(epochs, history['val_psnr'], label='Val PSNR')
    axes[1, 0].set_title('PSNR')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, history['train_l1'], label='Train L1')
    axes[1, 1].plot(epochs, history['val_l1'], label='Val L1')
    axes[1, 1].set_title('L1')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(save_path, artifact_path="visualizations")
    plt.close()
