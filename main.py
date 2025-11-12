import time
import json
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from autoencoder import StrainRateAutoencoder
from training import train_autoencoder
from inference import run_inference_with_metrics
from visualization import visualize_results, plot_training_history
from dataset import MemmapDataset


def main():
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 1000,
        
        'latent_dim': 2048,
        
        'mse_weight': 1,
        'l1_weight': 1,
        'fft_weight': 1e-7,
        
        'mixed_precision': True,
        'save_every': 200,
        'use_tensorboard': True,
        'project_name': 'ae-training-combined-loss',
        'run_name': f'combined_loss_run_{int(time.time())}',
    }

    mlflow.set_experiment(config['project_name'])
    with mlflow.start_run(run_name=config['run_name']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StrainRateAutoencoder(latent_dim=config['latent_dim'],
                                      use_batchnorm=False, dropout_rate=0.05)

        dataset = MemmapDataset("dataset_merged.npy", "dataset_stats.json", custom_size=20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        train_ds, val_ds = random_split(dataset, [n_train, n_total - n_train])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

        history = train_autoencoder(model, train_loader, val_loader, config, device)
        results = run_inference_with_metrics(model, train_loader, device)

        visualize_results(results[:3], 'autoencoder_results.png')
        plot_training_history(history, 'training_history.png')

        mlflow.log_metrics({
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
        })

        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact('training_history.json', artifact_path="history")


if __name__ == '__main__':
    main()
