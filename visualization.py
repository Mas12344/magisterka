import matplotlib.pyplot as plt
import mlflow
import numpy as np


def visualize_results(results, samples, save_path=None):
    n = len(results)
    

    n_samples = len(samples) if samples is not None else 0
    n_cols_samples = 4
    n_rows_samples = (n_samples + n_cols_samples - 1) // n_cols_samples if n_samples > 0 else 0
    

    total_rows = n + n_rows_samples
    
    fig = plt.figure(figsize=(20, 4 * total_rows))
    gs = fig.add_gridspec(total_rows, 6, hspace=0.3, wspace=0.3)
    

    for i, res in enumerate(results):
        inp = res['input'].squeeze()
        inp_aug = res['input_aug'].squeeze()
        out = res['output'].squeeze()
        out_aug = res['output_aug'].squeeze()
        diff = res['difference'].squeeze()
        
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(inp, cmap='viridis')
        ax0.set_title(f'Input {i+1}', fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(inp_aug, cmap='viridis')
        ax1.set_title(f'Input Aug {i+1}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(out, cmap='viridis')
        ax2.set_title(f'Reconstructed {i+1}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(out_aug, cmap='viridis')
        ax3.set_title(f'Reconstructed Aug {i+1}', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[i, 4])
        im1 = ax4.imshow(diff, cmap='hot', vmin=0)
        ax4.set_title(f'Diff {i+1}', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)
        
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
        
        ax5 = fig.add_subplot(gs[i, 5])
        ax5.text(
            0.05, 0.5, metrics_text,
            transform=ax5.transAxes,
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
        ax5.axis('off')
    

    if samples is not None and n_samples > 0:
        for idx, sample in enumerate(samples):
            row = n + idx // n_cols_samples
            col = idx % n_cols_samples
            
            ax_sample = fig.add_subplot(gs[row, col])
            sample_img = sample.squeeze()
            ax_sample.imshow(sample_img, cmap='viridis')
            ax_sample.set_title(f'Random Sample {idx+1}', fontsize=12, fontweight='bold')
            ax_sample.axis('off')
        

        for idx in range(n_samples, n_rows_samples * n_cols_samples):
            row = n + idx // n_cols_samples
            col = idx % n_cols_samples
            if col < 6:
                ax_empty = fig.add_subplot(gs[row, col])
                ax_empty.axis('off')
    
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
