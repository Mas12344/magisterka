import time
import json
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from autoencoder import StrainRateAutoencoder
from training import train_autoencoder, find_lr
from inference import run_inference_with_metrics
from visualization import visualize_results, plot_training_history
from dataset import *

from torch_lr_finder import TrainDataLoaderIter
class MyDataLoaderIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch):
        return batch, batch

def lr_range_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MemmapDataset("dataset_merged.npy", "dataset_stats.json", custom_size=10000)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    train_ds, val_ds = random_split(dataset, [n_train, n_total - n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    model = StrainRateAutoencoder(latent_dim=2048, use_batchnorm=False, dropout_rate=0.05)
    find_lr(model, MyDataLoaderIter(train_loader), device)


def main():
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'min_lr': 2e-8,
        'weight_decay': 0.01,
        'num_epochs': 50000,
        
        'latent_dim': 2048,
        
        'mse_weight': .1,
        'l1_weight': .1,
        'fft_weight': .3,
        
        'mixed_precision': False,
        'save_every': 10000,
        'use_tensorboard': True,
        'project_name': 'ae-training-combined-loss',
        'run_name': f'cosineannealing_scheduler_{int(time.time())}',
    }


    mlflow.set_experiment(config['project_name'])
    with mlflow.start_run(run_name=config['run_name']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = PickleDataset("../normalized_sorted.pkl", "../normalized_sorted_index.npy", custom_size=20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        train_ds, val_ds = random_split(dataset, [n_train, n_total - n_train])
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        model = StrainRateAutoencoder(latent_dim=config['latent_dim'],
                                      use_batchnorm=False, dropout_rate=0.05)

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


def search():
    import matplotlib.pyplot as plt 
    dataset = PickleDataset("../normalized_sorted.pkl", "../normalized_sorted_index.npy", custom_size=20)
    for i in range(20):
        sample = dataset[i]
        plt.imshow(sample.squeeze(0))
        plt.show()
        plt.close()


def inf_test():
    from autoencoder import combined_loss, calculate_metrics

    dataset = PickleDataset("../normalized_sorted.pkl", "../normalized_sorted_index.npy", custom_size=20)
    device = "cuda"
    model = StrainRateAutoencoder(latent_dim=2048, use_batchnorm=False, dropout_rate=0.05)

    state_dict = torch.load(r"I:\magisterka\clean\checkpoints\ae-training-combined-loss\cosineannealing_scheduler_1765106910\final_model.pth")['model_state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    results = []
    max_batches = 20

    with torch.no_grad():
        for i, batch in enumerate(dataset):
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

    visualize_results(results[:6], 'autoencoder_results.png')




if __name__ == '__main__':
    #main()
    #lr_range_test()
    #search()
    inf_test()
