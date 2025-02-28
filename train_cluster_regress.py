import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from scripts.network.network import GNSSRegressor  # 确保路径正确
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class GNSSDataset(Dataset):
    def __init__(self, features, labels, true_corrections, wls_ecef):
        self.features = features[:, :, :-1]  # Exclude the last column (cluster labels) from features
        self.labels = labels
        self.true_corrections = true_corrections
        self.wls_ecef = wls_ecef
        self.cluster_labels = features[:, 0, -1].astype(int)  # Use the first satellite's cluster label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
            torch.tensor(self.true_corrections[idx], dtype=torch.float),
            torch.tensor(self.wls_ecef[idx], dtype=torch.double),
            torch.tensor(self.cluster_labels[idx], dtype=torch.long)
        )

class GNSSLoss(nn.Module):
    def __init__(self, device=None):
        super(GNSSLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        if device:
            self.to(device)

    def forward(self, predictions, true_corrections):
        loss = self.mse_loss(predictions, true_corrections)
        return loss

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation MSE loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def plot_cdf(data, label, color):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    plt.plot(data_sorted, p, label=label, color=color)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Training'):
        optimizer.zero_grad()
        inputs, _, true_corrections, _, _ = batch
        inputs = inputs.to(device)
        true_corrections = true_corrections.to(device)
        #print(f"Train batch inputs shape: {inputs.shape}")
        predictions = model(inputs)
        loss = criterion(predictions, true_corrections)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions_list = []
    true_corrections_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            inputs, _, true_corrections, _, _ = batch
            inputs = inputs.to(device)
            true_corrections = true_corrections.to(device)
            #print(f"Validation batch inputs shape: {inputs.shape}")

            predictions = model(inputs)
            loss = criterion(predictions, true_corrections)

            total_loss += loss.item()

            predictions_list.append(predictions.cpu().numpy())
            true_corrections_list.append(true_corrections.cpu().numpy())

    return total_loss / len(loader), np.concatenate(predictions_list, axis=0), np.concatenate(true_corrections_list, axis=0)

def train_and_validate(config, train_loader, val_loader, cluster_id, writer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNSSRegressor(
        d_model=config.d_model,
        nhead=config.nhead,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_satellites=config.num_satellites,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = GNSSLoss(device=device).to(device)

    def lr_lambda(current_step: int):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        elif current_step < 100:
            return 0.999 ** (current_step - config.warmup_steps)
        else:
            return (0.998 ** (100 - config.warmup_steps)) * (0.998 ** (current_step - 100))

    scheduler = LambdaLR(optimizer, lr_lambda)

    early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=f'checkpoint_cluster_{cluster_id}.pt')

    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_loss, val_predictions, val_true_corrections = validate(model, val_loader, criterion, device)

        writer.add_scalar(f'Loss/Train_cluster_{cluster_id}', train_loss, epoch)
        writer.add_scalar(f'Loss/Validation_cluster_{cluster_id}', val_loss, epoch)
        writer.add_scalar(f'Learning_Rate_cluster_{cluster_id}', scheduler.get_last_lr()[0], epoch)

        print(
            f'Cluster {cluster_id} Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Cluster {cluster_id} Early stopping")
            break

    return model

def train_and_evaluate(config, train_dataset, val_dataset, writer):
    cluster_ids = np.unique(train_dataset.cluster_labels)
    models = {}

    for cluster_id in cluster_ids:
        print(f"Training model for cluster {cluster_id}")
        train_indices = (train_dataset.cluster_labels == cluster_id)
        val_indices = (val_dataset.cluster_labels == cluster_id)

        # Check if we have enough data for training and validation
        if np.sum(train_indices) == 0 or np.sum(val_indices) == 0:
            print(f"Skipping cluster {cluster_id} due to insufficient data")
            continue

        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, np.where(train_indices)[0]),
            batch_size=config[cluster_id].batch_size, shuffle=True, num_workers=config[cluster_id].num_workers)

        val_loader = DataLoader(
            torch.utils.data.Subset(val_dataset, np.where(val_indices)[0]),
            batch_size=config[cluster_id].batch_size, shuffle=False, num_workers=config[cluster_id].num_workers)

        # Print the shape of the data for debugging
        print(f"Cluster {cluster_id} train data shape: {train_loader.dataset[0][0].shape}")
        print(f"Cluster {cluster_id} validation data shape: {val_loader.dataset[0][0].shape}")

        models[cluster_id] = train_and_validate(config[cluster_id], train_loader, val_loader, cluster_id, writer)

    return models

def test(models, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_predictions = []
    all_true_corrections = []
    trace_predictions = []
    trace_true_corrections = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            inputs, _, true_corrections, _, cluster_labels = batch
            inputs = inputs.to(device)
            true_corrections = true_corrections.to(device)
            cluster_labels = cluster_labels.cpu().numpy()

            batch_predictions = np.zeros_like(true_corrections.cpu().numpy())

            for cluster_id in np.unique(cluster_labels):
                if cluster_id not in models:
                    print(f"Skipping cluster {cluster_id} due to missing model")
                    continue
                model = models[cluster_id]
                cluster_indices = (cluster_labels == cluster_id)
                cluster_inputs = inputs[cluster_indices]

                predictions = model(cluster_inputs).cpu().numpy()
                batch_predictions[cluster_indices] = predictions

            trace_predictions.append(batch_predictions)
            trace_true_corrections.append(true_corrections.cpu().numpy())

    return np.concatenate(trace_predictions, axis=0), np.concatenate(trace_true_corrections, axis=0)

def load_and_split_data_by_trace(root_dir):
    traces = {}
    for file in os.listdir(root_dir):
        if file.startswith('features_') and file.endswith('.npy'):
            trace_device_id = file.split('_', 1)[1].rsplit('.', 1)[0]
            trace_id = '_'.join(trace_device_id.split('_')[:-1])

            if trace_id not in traces:
                traces[trace_id] = {'features': [], 'labels': [], 'true_corrections': [], 'wls_ecef': []}

            features = np.load(os.path.join(root_dir, f'features_{trace_device_id}.npy'))
            labels = np.load(os.path.join(root_dir, f'labels_{trace_device_id}.npy'))
            true_corrections = np.load(os.path.join(root_dir, f'true_corrections_{trace_device_id}.npy'))
            wls_ecef = np.load(os.path.join(root_dir, f'wls_ecef_{trace_device_id}.npy'))

            # Ensure each feature block (81, 15) has a single cluster label
            cluster_label = features[:, 0, -1].astype(int)[0]  # Use the first satellite's cluster label
            features = features[:, :, :-1]  # Remove the last column
            features = np.concatenate([features, np.full((features.shape[0], features.shape[1], 1), cluster_label)], axis=-1)

            traces[trace_id]['features'].append(features)
            traces[trace_id]['labels'].append(labels)
            traces[trace_id]['true_corrections'].append(true_corrections)
            traces[trace_id]['wls_ecef'].append(wls_ecef)

    trace_ids = list(traces.keys())
    num_traces = len(trace_ids)
    val_size = int(0.2 * num_traces)
    train_size = num_traces - val_size

    np.random.shuffle(trace_ids)

    train_trace_ids = trace_ids[:train_size]
    val_trace_ids = trace_ids[train_size:]

    def concatenate_data(trace_ids):
        features = np.concatenate([np.concatenate(traces[trace_id]['features'], axis=0) for trace_id in trace_ids], axis=0)
        labels = np.concatenate([np.concatenate(traces[trace_id]['labels'], axis=0) for trace_id in trace_ids], axis=0)
        true_corrections = np.concatenate([np.concatenate(traces[trace_id]['true_corrections'], axis=0) for trace_id in trace_ids], axis=0)
        wls_ecef = np.concatenate([np.concatenate(traces[trace_id]['wls_ecef'], axis=0) for trace_id in trace_ids], axis=0)
        return features, labels, true_corrections, wls_ecef

    train_features, train_labels, train_true_corrections, train_wls_ecef = concatenate_data(train_trace_ids)
    val_features, val_labels, val_true_corrections, val_wls_ecef = concatenate_data(val_trace_ids)

    return train_features, train_labels, train_true_corrections, train_wls_ecef, val_features, val_labels, val_true_corrections, val_wls_ecef

def main():
    current_script_path = Path(__file__).resolve()
    root_path = current_script_path.parents[0]
    log_path = root_path / "logs" / "train_log"
    checkpoints_path = root_path / "checkpoints"

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = log_path / f'log_{current_time}'
    checkpoint_path = checkpoints_path / f'model_{current_time}'
    if os.path.exists(log_path):
        print('path exists')

    writer = SummaryWriter(str(log_dir))

    class Config:
        def __init__(self, d_model, nhead, d_ff, num_layers, num_satellites, batch_size, learning_rate, weight_decay, num_epochs, num_workers, patience, warmup_steps):
            self.d_model = d_model
            self.nhead = nhead
            self.d_ff = d_ff
            self.num_layers = num_layers
            self.num_satellites = num_satellites
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.num_epochs = num_epochs
            self.num_workers = num_workers
            self.patience = patience
            self.warmup_steps = warmup_steps

    # Define different configurations for different clusters
    configs = {
        0: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=1024, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=8000, num_workers=2, patience=10, warmup_steps=10),
        1: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=256, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=5000, num_workers=2, patience=10, warmup_steps=20),
        2: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=512, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=8000, num_workers=2, patience=10, warmup_steps=10),
        3: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=256, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=5000, num_workers=2, patience=10, warmup_steps=20),
        4: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=512, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=8000, num_workers=2, patience=10, warmup_steps=10),
        5: Config(d_model=32, nhead=4, d_ff=64, num_layers=3, num_satellites=81, batch_size=256, learning_rate=1e-5,
                  weight_decay=1e-4, num_epochs=5000, num_workers=2, patience=10, warmup_steps=20),
    }

    root_dir = r'D:\data\feature_cluster'

    train_features, train_labels, train_true_corrections, train_wls_ecef, val_features, val_labels, val_true_corrections, val_wls_ecef = load_and_split_data_by_trace(root_dir)

    train_dataset = GNSSDataset(train_features, train_labels, train_true_corrections, train_wls_ecef)
    val_dataset = GNSSDataset(val_features, val_labels, val_true_corrections, val_wls_ecef)

    models = train_and_evaluate(configs, train_dataset, val_dataset, writer)

    val_loader = DataLoader(val_dataset, batch_size=configs[0].batch_size, shuffle=False)
    test_predictions, test_true_corrections = test(models, val_loader)

    test_nn_errors = np.linalg.norm(test_predictions - test_true_corrections, axis=1)
    plt.figure()
    plot_cdf(test_nn_errors, 'Test NN Errors', 'green')
    plt.xlabel('Error (meters)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.title('CDF of Test Errors')
    plt.savefig(root_path / 'test_errors_cdf.png')
    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
