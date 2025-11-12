import matplotlib.pyplot as plt
import mlflow
import numpy as np


def visualize_results(results, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    axes = axes if n > 1 else axes.reshape(1, -1)

    for i, res in enumerate(results):
        inp, out, diff = res['input'].squeeze(), res['output'].squeeze(), res['difference'].squeeze()
        axes[i, 0].imshow(inp, cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(out, cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed {i+1}')
        axes[i, 1].axis('off')

        im = axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Abs Diff {i+1}')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2])

        m, l = res['metrics'], res['losses']
        text = f"""Metrics:
PSNR: {m['psnr']:.2f} dB
MAE: {m['mae']:.6f}

Losses:
MSE: {l['mse']:.6f}
L1: {l['l1']:.6f}
FFT: {l['fft']:.6f}
Total: {l['total']:.6f}"""
        axes[i, 3].text(0.1, 0.5, text, transform=axes[i, 3].transAxes,
                        fontsize=10, va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[i, 3].axis('off')

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
