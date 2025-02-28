#split BY trace and using cluster label as input
#data 8.2

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from scripts.network.network import GNSSClassifier
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import math


class GNSSDataset(Dataset):
    def __init__(self, features, labels, true_corrections, wls_ecef):
        self.features = features
        self.labels = labels
        self.true_corrections = true_corrections
        self.wls_ecef = wls_ecef

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
            torch.tensor(self.true_corrections[idx], dtype=torch.float),
            torch.tensor(self.wls_ecef[idx], dtype=torch.double)
        )

class GNSSLoss(nn.Module):
    def __init__(self, alpha, num_classes=12, error_range=(-25, 25), device=None):
        super(GNSSLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.error_range = error_range
        self.error_values = torch.tensor(
            [-25, -15, -10, -5, -2, -1, 1, 2, 5, 10, 15, 25], dtype=torch.float)
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        if device:
            self.error_values = self.error_values.to(device)

    def forward(self, x_out, y_out, z_out, labels, true_corrections):
        celoss_x = F.cross_entropy(x_out, labels[:, :, 0].argmax(dim=1))
        celoss_y = F.cross_entropy(y_out, labels[:, :, 1].argmax(dim=1))
        celoss_z = F.cross_entropy(z_out, labels[:, :, 2].argmax(dim=1))
        celoss = celoss_x + celoss_y + celoss_z

        mse_x = self.calculate_mse(x_out, true_corrections[:, 0])
        mse_y = self.calculate_mse(y_out, true_corrections[:, 1])
        mse_z = self.calculate_mse(z_out, true_corrections[:, 2])
        mseloss = mse_x + mse_y + mse_z

        loss = mseloss+celoss
        return loss, celoss, mseloss, celoss_x, celoss_y, celoss_z, mse_x, mse_y, mse_z

    def calculate_mse(self, outputs, true_correction):
        predictions = torch.matmul(F.softmax(outputs, dim=1),
                                   self.error_values.to(outputs.device).unsqueeze(1)).squeeze(1)
        mse = F.mse_loss(predictions, true_correction)
        return mse

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

def calculate_wls_mse(dataset):
    total_wls_mse = 0.0
    total_count = 0
    wls_errors = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        _, _, true_corrections, wls_ecef = data
        true_corrections = true_corrections.to(torch.double)
        wls_ecef = wls_ecef.to(torch.double)

        wls_mse = F.mse_loss(wls_ecef + true_corrections, wls_ecef, reduction='sum').item()
        total_wls_mse += wls_mse
        total_count += 1
        wls_errors.extend((wls_ecef + true_corrections - wls_ecef).cpu().numpy())

    average_wls_mse = total_wls_mse / total_count if total_count > 0 else float('nan')
    return average_wls_mse, np.array(wls_errors)

def plot_cdf(data, label, color):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    plt.plot(data_sorted, p, label=label, color=color)

def main():
    global train_loss, val_loss
    current_script_path = Path(__file__).resolve()
    root_path = current_script_path.parents[0]
    processed_path = root_path / "data" / "processed_data"
    log_path = root_path / "logs" / "train_log"
    checkpoints_path = root_path / "checkpoints"

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = log_path / f'log_{current_time}'
    checkpoint_path = checkpoints_path / f'model_{current_time}'
    if os.path.exists(log_path):
        print('path exists')

    writer = SummaryWriter(str(log_dir))

    class Config:
        d_model = 64
        nhead = 4
        d_ff = 64
        num_layers = 4
        num_satellites = 81
        num_classes = 12
        batch_size = 4096
        learning_rate = 1e-5
        weight_decay = 1e-4 
        num_epochs = 2000
        num_workers = 2
        alpha = 0.1
        patience = 800  # Add patience for early stopping
        use_amp = True

    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = r'D:\data\feature_cluster'

    # Read all traces and devices
    traces = {}
    for file in os.listdir(root_dir):
        if file.startswith('features_') and file.endswith('.npy'):
            full_id = file[len('features_'):-len('.npy')]  # 提取完整ID
            trace_id = '_'.join(full_id.split('_')[:-1])  # 提取trace部分，去掉设备ID
            if trace_id not in traces:
                traces[trace_id] = {'features': [], 'labels': [], 'true_corrections': [], 'wls_ecef': []}

            traces[trace_id]['features'].append(np.load(os.path.join(root_dir, f'features_{full_id}.npy')))
            traces[trace_id]['labels'].append(np.load(os.path.join(root_dir, f'labels_{full_id}.npy')))
            traces[trace_id]['true_corrections'].append(np.load(os.path.join(root_dir, f'true_corrections_{full_id}.npy')))
            traces[trace_id]['wls_ecef'].append(np.load(os.path.join(root_dir, f'wls_ecef_{full_id}.npy')))

    trace_ids = list(traces.keys())
    num_traces = len(trace_ids)
    val_size = int(0.35 * num_traces)
    train_size = num_traces - val_size

    np.random.shuffle(trace_ids)

    train_trace_ids = trace_ids[:train_size]
    val_trace_ids = trace_ids[train_size:]

    train_features = np.concatenate([np.concatenate(traces[trace_id]['features'], axis=0) for trace_id in train_trace_ids], axis=0)
    train_labels = np.concatenate([np.concatenate(traces[trace_id]['labels'], axis=0) for trace_id in train_trace_ids], axis=0)
    train_true_corrections = np.concatenate([np.concatenate(traces[trace_id]['true_corrections'], axis=0) for trace_id in train_trace_ids], axis=0)
    train_wls_ecef = np.concatenate([np.concatenate(traces[trace_id]['wls_ecef'], axis=0) for trace_id in train_trace_ids], axis=0)

    val_features = np.concatenate([np.concatenate(traces[trace_id]['features'], axis=0) for trace_id in val_trace_ids], axis=0)
    val_labels = np.concatenate([np.concatenate(traces[trace_id]['labels'], axis=0) for trace_id in val_trace_ids], axis=0)
    val_true_corrections = np.concatenate([np.concatenate(traces[trace_id]['true_corrections'], axis=0) for trace_id in val_trace_ids], axis=0)
    val_wls_ecef = np.concatenate([np.concatenate(traces[trace_id]['wls_ecef'], axis=0) for trace_id in val_trace_ids], axis=0)

    train_dataset = GNSSDataset(train_features, train_labels, train_true_corrections, train_wls_ecef)
    val_dataset = GNSSDataset(val_features, val_labels, val_true_corrections, val_wls_ecef)

    # Log train and validation trace-device IDs
    with open(root_path / f'train_val_trace_ids_{current_time}.log', 'w') as log_file:
        log_file.write("Training set trace IDs:\n")
        for trace_id in train_trace_ids:
            log_file.write(f"{trace_id}\n")

        log_file.write("\nValidation set trace IDs:\n")
        for trace_id in val_trace_ids:
            log_file.write(f"{trace_id}\n")

    # Calculate WLS MSE for train and validation datasets
    train_wls_mse, train_wls_errors = calculate_wls_mse(train_dataset)
    val_wls_mse, val_wls_errors = calculate_wls_mse(val_dataset)

    print(f'Train WLS MSE: {train_wls_mse:.4f}')
    print(f'Validation WLS MSE: {val_wls_mse:.4f}')

    # # Plot CDF for train and validation WLS errors
    # plt.figure()
    # plot_cdf(np.linalg.norm(train_wls_errors, axis=1), 'Train WLS Errors', 'blue')
    # plot_cdf(np.linalg.norm(val_wls_errors, axis=1), 'Validation WLS Errors', 'red')
    # plt.xlabel('Error (meters)')
    # plt.ylabel('CDF')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(root_path / 'wls_errors_cdf.png')
    # plt.show()

    model = GNSSClassifier(
        d_model=config.d_model,
        nhead=config.nhead,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_satellites=config.num_satellites,
        num_classes=config.num_classes
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model = model.to(device)
    criterion = GNSSLoss(alpha=config.alpha, device=device).to(device)

    warmup_steps = 20
    total_steps = len(train_loader) * config.num_epochs

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step <= 100:
            return 0.998 ** (current_step - warmup_steps)
        else:
            return (0.998 ** (100 - warmup_steps)) * (0.998 ** (current_step - 100))
    # def lr_lambda(current_step: int):
    #     if current_step < warmup_steps:
    #         # 线性预热
    #         return float(current_step) / float(max(1, warmup_steps))
    #     else:
    #         # 余弦退火衰减
    #         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    #         return 0.5 * (1.0 + math.cos(math.pi * progress))


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=checkpoint_path)

    def train(model, loader, optimizer, criterion, device):
        model.train()
        total_loss, total_celoss, total_mseloss = 0, 0, 0
        for batch in tqdm(loader, desc='Training'):
            optimizer.zero_grad()
            inputs, labels, true_corrections, wls_ecef = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_corrections = true_corrections.to(device)
            wls_ecef = wls_ecef.to(device)
            x_out, y_out, z_out = model(inputs)
            loss, celoss, mseloss, celoss_x, celoss_y, celoss_z, mseloss_x, mseloss_y, mseloss_z = criterion(
                x_out, y_out, z_out, labels, true_corrections)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_celoss += celoss.item()
            total_mseloss += mseloss.item()

        return total_loss / len(loader), total_celoss / len(loader), total_mseloss / len(loader)

    def validate(model, loader, criterion, device):
        model.eval()
        total_loss, total_celoss, total_mseloss = 0, 0, 0
        val_predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validation'):
                inputs, labels, true_corrections, wls_ecef = batch
                inputs, labels, true_corrections = inputs.to(device), labels.to(device), true_corrections.to(device)
                wls_ecef = wls_ecef.to(device)

                x_out, y_out, z_out = model(inputs)
                loss, celoss, mseloss, celoss_x, celoss_y, celoss_z, mseloss_x, mseloss_y, mseloss_z = criterion(
                    x_out, y_out, z_out, labels, true_corrections)

                total_loss += loss.item()
                total_celoss += celoss.item()
                total_mseloss += mseloss.item()

                val_predictions.append(torch.cat([x_out, y_out, z_out], dim=1).cpu().numpy())

        return total_loss / len(loader), total_celoss / len(loader), total_mseloss / len(loader), np.concatenate(val_predictions, axis=0)

    def calculate_predictions(predictions, criterion, device):
        predicted_corrections = []
        for i in range(0, predictions.shape[1], criterion.num_classes):
            pred_tensor = torch.tensor(predictions[:, i:i + criterion.num_classes]).to(device)
            softmax_tensor = F.softmax(pred_tensor, dim=1)
            correction = torch.matmul(softmax_tensor, criterion.error_values.unsqueeze(1).to(device)).squeeze(1)
            predicted_corrections.append(correction.cpu().numpy())
        result = np.column_stack(predicted_corrections)
        print("Predicted Corrections shape:", result.shape)
        return result

    try:
        for epoch in range(config.num_epochs):
            train_loss, train_celoss, train_mseloss = train(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_loss, val_celoss, val_mseloss, val_predictions = validate(model, val_loader, criterion, device)

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('CELoss/Train', train_celoss, epoch)
            writer.add_scalar('CELoss/Validation', val_celoss, epoch)
            writer.add_scalar('MSELoss/Train', train_mseloss, epoch)
            writer.add_scalar('MSELoss/Validation', val_mseloss, epoch)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

            print(
                f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train MSE: {train_mseloss:.4f}, Validation MSE: {val_mseloss:.4f}')

            early_stopping(val_mseloss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Calculate predictions and plot CDF for validation set

        val_predictions = calculate_predictions(val_predictions, criterion, device)
        val_nn_errors = np.linalg.norm(val_predictions - val_true_corrections, axis=1)

        plt.figure()
        plot_cdf(np.linalg.norm(val_wls_errors, axis=1), 'Validation WLS Errors', 'red')
        plot_cdf(val_nn_errors, 'Validation NN Errors', 'green')
        plt.xlabel('Error (meters)')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.savefig(root_path / 'combined_cdf.png')
        plt.show()

    finally:
        writer.add_hparams(hparam_dict={
            'd_model': config.d_model,
            'nhead': config.nhead,
            'd_ff': config.d_ff,
            'num_layers': config.num_layers,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'alpha': config.alpha,
        },
            metric_dict={
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
        )
        writer.flush()
        writer.close()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
