'''
    4.29 修改输入大小与padding，变成按星座种类排序
'''
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ADR_STATE_UNKNOWN = 0
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2
ADR_STATE_HALF_CYCLE_RESOLVED = 1 << 3
ADR_STATE_HALF_CYCLE_REPORTED = 1 << 4


def get_state_weight(state):
    weight = 0
    if state & ADR_STATE_VALID:
        weight += 32  # VALID 状态权重最高
    if state & ADR_STATE_HALF_CYCLE_RESOLVED:
        weight += 16  # HALF_CYCLE_RESOLVED 次之
    if state & ADR_STATE_HALF_CYCLE_REPORTED:
        weight += 8  # HALF_CYCLE_REPORTED 次之
    # 未知或其他状态不加分
    return weight


class GNSSDataset(Dataset):
    def __init__(self, root_dir, num_satellites=60):
        self.root_dir = Path(root_dir)
        self.num_satellites = num_satellites
        self.signal_max_satellites = {
            'GPS_L1_CA': 16, 'GLO_G1_CA': 14, 'BDS_B1_I': 12,
            'GAL_E1_C_P': 12, 'GAL_E5A_Q': 11, 'GPS_L5_Q': 8, 'BDS_B2A_P': 8
        }
        # 定义信号类型的排序优先级
        self.signal_order = {
            'GPS_L1_CA': 1, 'GLO_G1_CA': 2, 'BDS_B1_I': 3,
            'GAL_E1_C_P': 4, 'GAL_E5A_Q': 5, 'GPS_L5_Q': 6, 'BDS_B2A_P': 7
        }
        self.data, self.labels, self.true_corrections, self.wls_ecef = self._load_data()

    def _correction_to_one_hots(self, correction):
        error = np.clip(np.round(correction), -10, 10)
        one_hot_vector = np.zeros(21)
        index = int(error + 10)
        one_hot_vector[index] = 1
        return one_hot_vector

    def _correction_to_one_hot(self,correction):
        one_hot_vector = np.zeros(22)
        if correction < -20:
            index = 0
        elif -20 <= correction < -13:
            index = 1
        elif -13 <= correction < -8:
            index = 2
        elif -8 <= correction < 0:  # 负数：左闭右开
            index = int(np.floor(correction + 8)) + 3
            if index < 11:  # 确保索引不超出范围
                one_hot_vector[index + 1] = 1
        elif 0 < correction <= 8:
            index = int(np.ceil(correction + 8)) + 2
            if index > 11:  # 确保索引不超出范围
                one_hot_vector[index - 1] = 1
        elif 8 < correction <= 13:
            index = 19
        elif 13 < correction <= 20:
            index = 20
        elif correction > 20:
            index = 21
        elif correction == 0:  # 对0的处理
            index = 11

        one_hot_vector[index] = 1
        return one_hot_vector

    def _calculate_distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def _load_data(self):
        exclude_traces = [
            '2022-02-24-18-29-us-ca-lax-o/pixel5',
            '2021-01-04-21-50-us-ca-e1highway280driveroutea/pixel4',
            '2023-09-05-20-13-us-ca/pixel7pro',
            '2020-08-04-00-19-us-ca-sb-mtv-101/pixel4',
            '2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl',
            '2023-09-06-18-47-us-ca/pixel6pro',
            '2020-12-10-22-52-us-ca-sjc-c/pixel5',

            '2022-02-24-18-29-us-ca-lax-o/mi8',
            '2021-03-16-20-40-us-ca-mtv-b/mi8',
            '2023-09-07-22-47-us-ca-routebc2/pixel6pro',
            '2022-04-01-18-22-us-ca-lax-t/mi8',
            '2022-05-13-20-57-us-ca-mtv-pe1/sm-g988b',
            '2020-12-10-22-17-us-ca-sjc-c/mi8',
            '2022-01-26-20-02-us-ca-mtv-pe1/sm-g988b',
            '2022-01-26-20-02-us-ca-mtvpe1/mi8',
            '2021-08-24-20-32-us-ca-mtv-h/mi8',
            '2022-08-04-20-07-us-ca-sjc-q/mi8',
            '2021-12-07-19-22-us-ca-lax-d/mi8',
            '2021-12-08-17-22-us-ca-lax-a/pixel6pro',
            '2023-09-06-18-04-us-ca/sm-s908b'
        ]
        all_features = []
        all_labels = []
        all_true_corrections = []
        all_wls_ecef = []
        exclude_traces = [os.path.normpath(trace) for trace in exclude_traces]
        for subdir, dirs, files in os.walk(self.root_dir):
            norm_subdir = os.path.normpath(subdir)
            should_exclude = any(exclude_trace in norm_subdir for exclude_trace in exclude_traces)
            if should_exclude:
                print("passing", subdir)
                continue
            gnss_file = os.path.join(subdir, 'gnss_data.csv')
            ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

            if os.path.exists(gnss_file) and os.path.exists(ground_truth_file):
                gnss_df = pd.read_csv(gnss_file)
                ground_truth_df = pd.read_csv(ground_truth_file)
                if ground_truth_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters',
                          'WlsPositionXEcefMeters']].isnull().any().any():
                    continue
                columns_to_remove = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
                gnss_df = gnss_df.drop(columns=columns_to_remove)
                merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')
                unique_timestamps = merged_df['utcTimeMillis'].unique()

                for timestamp in unique_timestamps:
                    timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp].copy()
                    timestamp_data = timestamp_data[~timestamp_data['SignalType'].isin(['QZS_L1_CA', 'QZS_L5_Q'])]
                    wls_ecef_coords = \
                    timestamp_data[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].iloc[
                        0].to_numpy()
                    all_wls_ecef.append(wls_ecef_coords)

                    # 计算每个卫星的ADR状态权重并赋值
                    timestamp_data.loc[:, 'StateWeight'] = timestamp_data['AccumulatedDeltaRangeState'].apply(
                        get_state_weight)

                    # 按照仰角、SVID和ADR状态权重进行排序
                    timestamp_data['Order'] = timestamp_data['SignalType'].map(self.signal_order)
                    timestamp_data.sort_values(by=['Order', 'StateWeight', 'SvElevationDegrees'],
                                               ascending=[True, False, False], inplace=True)
                    signal_types = timestamp_data['SignalType'].unique()
                    features = []

                    pseudorange_residuals = []
                    los_vectors = []
                    for index, row in timestamp_data.iterrows():
                        if 'true_correction_x' not in timestamp_data.columns:
                            continue
                        #print(timestamp_data.head())
                        true_correction = np.array([
                            timestamp_data['true_correction_x'].iloc[0],
                            timestamp_data['true_correction_y'].iloc[0],
                            timestamp_data['true_correction_z'].iloc[0]
                        ])
                        if np.any(np.abs(true_correction) > 35):
                            continue
                        distance = self._calculate_distance(
                            row['WlsPositionXEcefMeters'], row['WlsPositionYEcefMeters'], row['WlsPositionZEcefMeters'],
                            row['SvPositionXEcefMeters'], row['SvPositionYEcefMeters'], row['SvPositionZEcefMeters']
                        )

                        residual = row['RawPseudorangeMeters'] - distance
                        pseudorange_residuals.append(residual)

                        los_vector = np.array([
                            row['SvPositionXEcefMeters'] - row['WlsPositionXEcefMeters'],
                            row['SvPositionYEcefMeters'] - row['WlsPositionYEcefMeters'],
                            row['SvPositionZEcefMeters'] - row['WlsPositionZEcefMeters']
                        ])
                        # Normalize the los_vector
                        los_vector_normalized = los_vector / np.linalg.norm(los_vector)
                        los_vectors.append(los_vector_normalized)

                    pseudorange_residuals = np.array(pseudorange_residuals)
                    if len(los_vectors) == 0:
                        continue
                    los_vectors = np.stack(los_vectors)

                    gnss_features = timestamp_data[[
                        'Cn0DbHz', 'IonosphericDelayMeters', 'TroposphericDelayMeters',
                        'SvElevationDegrees', 'SvAzimuthDegrees',
                        'PseudorangeRateMetersPerSecond',
                        'PseudorangeRateUncertaintyMetersPerSecond',
                        'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                        'SvVelocityZEcefMetersPerSecond',
                        'SvClockBiasMeters'
                    ]].fillna(0).to_numpy()

                    # Normalize
                    scaler_velocity = StandardScaler()
                    gnss_features[:, 7:10] = scaler_velocity.fit_transform(gnss_features[:, 7:10])

                    # Normalize SvClockBiasMeters
                    scaler_clock_bias = StandardScaler()
                    gnss_features[:, 10] = scaler_clock_bias.fit_transform(
                        gnss_features[:, 10].reshape(-1, 1)).flatten()

                    gnss_features = np.hstack([gnss_features, los_vectors, pseudorange_residuals[:, np.newaxis]])
                    #print(gnss_features.shape)

                    # Pad features if necessary
                    # if gnss_features.shape[0] < self.num_satellites:
                    #     padding = np.zeros((self.num_satellites - gnss_features.shape[0], gnss_features.shape[1]))
                    #     gnss_features = np.vstack([gnss_features, padding])
                    combined_features = None
                    for signal_type in self.signal_max_satellites:
                        # 找出当前信号类型的所有特征行
                        type_mask = (timestamp_data['SignalType'] == signal_type)
                        type_features = gnss_features[type_mask]
                        #print(type_features.shape,"当前星座数量")

                        # 计算需要填充的行数
                        max_count = self.signal_max_satellites[signal_type]
                        current_count = type_features.shape[0]
                        if current_count < max_count:
                            # 如果当前行数小于最大行数，进行填充
                            padding_size = max_count - current_count
                            padding = np.zeros((padding_size, gnss_features.shape[1]))
                            type_features = np.vstack([type_features, padding])  # 将填充添加到特征行
                            #print(type_features.shape,"填充后大小")
                        if combined_features is None:
                            combined_features = type_features
                        else:
                            combined_features = np.vstack([combined_features, type_features])


                    #print(combined_features.shape,"合并后大小")
                    # Convert corrections to one-hot vectors
                    correction_x = self._correction_to_one_hot(timestamp_data['true_correction_x'].iloc[0])
                    correction_y = self._correction_to_one_hot(timestamp_data['true_correction_y'].iloc[0])
                    correction_z = self._correction_to_one_hot(timestamp_data['true_correction_z'].iloc[0])

                    all_features.append(combined_features)
                    all_labels.append(np.stack([correction_x, correction_y, correction_z], axis=-1))
                    all_true_corrections.append(true_correction)  # 添加true_correction用与计算mse

        return np.array(all_features), np.array(all_labels), np.array(all_true_corrections), np.array(all_wls_ecef)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
            torch.tensor(self.true_corrections[idx], dtype=torch.float),  # 返回true_corrections
            torch.tensor(self.wls_ecef[idx], dtype=torch.double)
        )


#Usage example
# root_dir = r'C:\Users\BD\Desktop\gnss\data\processed_data\2020-08-13-21-41-us-ca-mtv-sf-280'
#
# dataset = GNSSDataset(root_dir)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# for features, labels, ture_correct, wls in dataloader:
#     #print(features)
#     #print(labels)
#     print(wls)
#     print(ture_correct)
#     print(f"Batch features shape: {features.shape}")  # 打印特征的形状
#     print(f"Batch labels shape: {labels.shape}")  # 打印标签的形状
#     print(f"Batch true_correct shape: {ture_correct.shape}")  # 打印true_correct的形状
#     print(f"Batch wls shape: {wls.shape}")  # 打印wls的形状
#
#     break