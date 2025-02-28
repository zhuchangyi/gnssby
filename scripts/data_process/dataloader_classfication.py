import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class GNSSDataset(Dataset):
    def __init__(self, root_dir, num_satellites=20):
        self.root_dir = Path(root_dir)
        self.num_satellites = num_satellites
        self.data, self.labels = self._load_data()

    def _correction_to_one_hot(self, correction):
        error = np.clip(np.round(correction), -10, 10)
        one_hot_vector = np.zeros(21)
        index = int(error + 10)
        one_hot_vector[index] = 1
        return one_hot_vector

    def _calculate_distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def _load_data(self):
        all_features = []
        all_labels = []
        for subdir, dirs, files in os.walk(self.root_dir):
            gnss_file = os.path.join(subdir, 'gnss_data.csv')
            ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

            if os.path.exists(gnss_file) and os.path.exists(ground_truth_file):
                gnss_df = pd.read_csv(gnss_file)
                ground_truth_df = pd.read_csv(ground_truth_file)

                merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')
                unique_timestamps = merged_df['utcTimeMillis'].unique()

                for timestamp in unique_timestamps:
                    timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp]

                    correction_x = self._correction_to_one_hot(timestamp_data['true_correction_x'].iloc[0])
                    correction_y = self._correction_to_one_hot(timestamp_data['true_correction_y'].iloc[0])
                    correction_z = self._correction_to_one_hot(timestamp_data['true_correction_z'].iloc[0])

                    all_features.append(timestamp_data.drop(
                        columns=['utcTimeMillis', 'true_correction_x', 'true_correction_y',
                                 'true_correction_z']).fillna(0).to_numpy()[:self.num_satellites])
                    all_labels.append(np.stack([correction_x, correction_y, correction_z]))

        # Pad features to ensure all have the same shape
        padded_features = np.zeros((len(all_features), self.num_satellites, all_features[0].shape[1]))
        for i, features in enumerate(all_features):
            length = min(self.num_satellites, features.shape[0])
            padded_features[i, :length, :] = features[:length, :]

        return padded_features, np.array(all_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


# Usage
# Assuming your data is located under 'data/processed'
data_root_dir = './data/processed_data/2020-06-25-00-34-us-ca-mtv-sb-101'

dataset = GNSSDataset(data_root_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for features, labels in dataloader:
    print(
        f"Features shape: {features.shape}")
    print(
        f"Labels shape: {labels.shape}")
    # Here goes your model training code
