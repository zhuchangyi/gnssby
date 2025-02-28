import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import logging
from collections import Counter
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gnss_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def detect_acceleration_options():
    """
    检测可用的加速选项
    """
    acceleration_options = {
        'gpu': False,
        'intel': False,
        'cpu_cores': psutil.cpu_count(logical=False)
    }

    # 检测PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            acceleration_options['gpu'] = True
            logger.info(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        else:
            logger.info("未检测到支持的GPU或CUDA环境")
    except ImportError:
        logger.info("未安装PyTorch，GPU加速不可用")

    # 检测Intel扩展
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        acceleration_options['intel'] = True
        logger.info("检测到Intel CPU，已启用Intel加速")
    except ImportError:
        logger.info("未检测到Intel扩展库")

    return acceleration_options


class GNSSClusteringModel:
    def __init__(self, data_path, save_path, scaler_type='robust'):
        """
        初始化GNSS聚类模型

        参数:
            data_path: 数据路径
            save_path: 结果保存路径
            scaler_type: 特征缩放方法，可选'robust', 'standard', 'minmax'
        """
        self.dbscan_labels = None  # 存储DBSCAN标签
        self.kmeans_labels = None  # 存储K-Means标签
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.scaler_type = scaler_type
        self.scaler = None
        self.features = None
        self.timestamps = None
        self.locations = None  # 对应位置信息
        self.cluster_labels = None
        self.best_model = None
        self.features_df = None

        # 检测可用的加速选项
        self.acceleration = detect_acceleration_options()

    def _process_single_file(self, file_path):
        """
        处理单个GNSS数据文件
        """
        file_data = []
        file_locations = []
        file_timestamps = []

        try:
            # 加载GNSS数据
            gnss_df = pd.read_csv(file_path)

            # 检查是否包含必要的列
            required_columns = ['utcTimeMillis', 'SignalType', 'Cn0DbHz', 'SvElevationDegrees',
                                'SvAzimuthDegrees', 'RawPseudorangeMeters']
            missing_columns = [col for col in required_columns if col not in gnss_df.columns]
            if missing_columns:
                return [], [], []

            # 加载对应的地面真值数据
            ground_truth_file = file_path.parent / "ground_truth.csv"
            if not ground_truth_file.exists():
                return [], [], []

            ground_truth_df = pd.read_csv(ground_truth_file)

            # 检查地面真值文件是否包含必要的列
            gt_required_columns = ['utcTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees',
                                   'AltitudeMeters', 'WlsPositionXEcefMeters',
                                   'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
            gt_missing_columns = [col for col in gt_required_columns if col not in ground_truth_df.columns]
            if gt_missing_columns:
                return [], [], []

            # 检查地面真值数据是否有缺失值
            if ground_truth_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters',
                                'WlsPositionXEcefMeters']].isnull().any().any():
                return [], [], []

            # 避免列名冲突
            duplicate_columns = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
            gnss_df = gnss_df.drop(columns=duplicate_columns, errors='ignore')

            # 合并数据
            merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')

            # 记录处理的trace和device ID
            trace_id = file_path.parent.parent.name
            device_id = file_path.parent.name

            # 对每个时间戳的数据进行处理
            for timestamp in merged_df['utcTimeMillis'].unique():
                timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp].copy()

                # 只保留GPS_L1_CA和BDS_B1_I信号
                timestamp_data = timestamp_data[
                    timestamp_data['SignalType'].isin(['GPS_L1_CA', 'BDS_B1_I'])]

                # 跳过没有足够卫星的时间戳
                if len(timestamp_data) < 4:  # 至少需要4颗卫星
                    continue

                # 提取特征
                features_dict = self._extract_features(timestamp_data)
                if features_dict:
                    file_data.append(features_dict)

                    # 保存位置信息
                    location = {
                        'lat': timestamp_data['LatitudeDegrees'].iloc[0],
                        'lon': timestamp_data['LongitudeDegrees'].iloc[0],
                        'alt': timestamp_data['AltitudeMeters'].iloc[0],
                        'wls_x': timestamp_data['WlsPositionXEcefMeters'].iloc[0],
                        'wls_y': timestamp_data['WlsPositionYEcefMeters'].iloc[0],
                        'wls_z': timestamp_data['WlsPositionZEcefMeters'].iloc[0],
                        'trace_id': trace_id,
                        'device_id': device_id
                    }
                    file_locations.append(location)
                    file_timestamps.append(timestamp)

            return file_data, file_locations, file_timestamps
        except Exception as e:
            return [], [], []

    def load_data(self, file_pattern="**/*.csv", max_workers=None):
        """
        使用并行处理加载GNSS数据文件

        参数:
            file_pattern: 文件匹配模式，默认为"**/*.csv"以递归搜索所有CSV文件
            max_workers: 并行进程数，默认为CPU核心数的75%
        """
        logger.info(f"加载数据文件...")
        all_data = []
        all_locations = []
        all_timestamps = []

        # 验证数据路径是否存在
        if not self.data_path.exists():
            logger.error(f"数据路径不存在: {self.data_path}")
            return False

        # 递归搜索所有gnss_data.csv文件
        gnss_files = list(self.data_path.glob("**/gnss_data.csv"))
        logger.info(f"找到 {len(gnss_files)} 个GNSS数据文件")

        if len(gnss_files) == 0:
            logger.error(f"在 {self.data_path} 中未找到任何gnss_data.csv文件")
            return False

        # 确定进程数量，默认为CPU核心数的75%
        if max_workers is None:
            max_workers = max(1, int(self.acceleration['cpu_cores'] * 0.75))

        logger.info(f"使用 {max_workers} 个进程并行处理文件")

        # 并行处理文件
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self._process_single_file, gnss_files), total=len(gnss_files)))

        # 处理结果
        for file_data, file_locations, file_timestamps in results:
            if file_data:
                all_data.extend(file_data)
                all_locations.extend(file_locations)
                all_timestamps.extend(file_timestamps)

        if not all_data:
            logger.error("没有加载到有效数据!")
            return False

        # 将特征列表转换为DataFrame
        self.features_df = pd.DataFrame(all_data)
        self.locations = pd.DataFrame(all_locations)
        self.timestamps = np.array(all_timestamps)

        # 处理缺失值
        self.features_df = self.features_df.fillna(self.features_df.median())

        logger.info(f"成功加载 {len(self.features_df)} 个时间点的数据")
        if 'trace_id' in self.locations.columns and 'device_id' in self.locations.columns:
            logger.info(f"来自 {self.locations['trace_id'].nunique()} 个轨迹和 "
                        f"{self.locations['device_id'].nunique()} 个设备")

        return True

    def _extract_features(self, timestamp_data):
        """
        从时间戳数据中提取聚类所需的特征
        """
        try:
            # 1. 卫星数量特征
            num_satellites = len(timestamp_data)

            # 按星座类型统计卫星数量
            gps_count = (timestamp_data['SignalType'] == 'GPS_L1_CA').sum()
            bds_count = (timestamp_data['SignalType'] == 'BDS_B1_I').sum()
            gps_ratio = gps_count / num_satellites if num_satellites > 0 else 0

            # 2. 信号质量特征
            cn0_mean = timestamp_data['Cn0DbHz'].mean()
            cn0_std = timestamp_data['Cn0DbHz'].std()
            cn0_min = timestamp_data['Cn0DbHz'].min()
            cn0_max = timestamp_data['Cn0DbHz'].max()

            # 3. 卫星几何分布特征
            # 仰角特征
            elevation_mean = timestamp_data['SvElevationDegrees'].mean()
            elevation_std = timestamp_data['SvElevationDegrees'].std()
            elevation_min = timestamp_data['SvElevationDegrees'].min()
            low_elevation_count = (timestamp_data['SvElevationDegrees'] < 15).sum()
            high_elevation_count = (timestamp_data['SvElevationDegrees'] > 60).sum()

            # 方位角特征（转换为正弦余弦表示）
            sin_azimuth = np.sin(np.radians(timestamp_data['SvAzimuthDegrees'])).mean()
            cos_azimuth = np.cos(np.radians(timestamp_data['SvAzimuthDegrees'])).mean()
            azimuth_std = timestamp_data['SvAzimuthDegrees'].std()

            # 4. 伪距残差特征
            # 计算伪距残差
            pseudorange_residuals = []
            for _, row in timestamp_data.iterrows():
                try:
                    # 计算卫星到接收机的理论距离
                    dx = row['SvPositionXEcefMeters'] - row['WlsPositionXEcefMeters']
                    dy = row['SvPositionYEcefMeters'] - row['WlsPositionYEcefMeters']
                    dz = row['SvPositionZEcefMeters'] - row['WlsPositionZEcefMeters']
                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # 伪距残差 = 测量的伪距 - 理论距离
                    residual = row['RawPseudorangeMeters'] - distance
                    pseudorange_residuals.append(residual)
                except:
                    continue

            if pseudorange_residuals:
                residual_mean = np.mean(pseudorange_residuals)
                residual_std = np.std(pseudorange_residuals)
                residual_max = np.max(np.abs(pseudorange_residuals))
            else:
                residual_mean = residual_std = residual_max = np.nan

            # 5. 延迟相关特征
            iono_delay_mean = timestamp_data['IonosphericDelayMeters'].mean()
            iono_delay_std = timestamp_data['IonosphericDelayMeters'].std()
            tropo_delay_mean = timestamp_data['TroposphericDelayMeters'].mean()
            tropo_delay_std = timestamp_data['TroposphericDelayMeters'].std()

            # 6. ADR状态特征
            # 计算有效的ADR测量比例
            adr_valid_ratio = 0
            try:
                adr_valid = ((timestamp_data['AccumulatedDeltaRangeState'] & 1) > 0).sum()
                adr_valid_ratio = adr_valid / num_satellites
            except:
                pass

            # 7. 卫星速度和伪距率特征
            pr_rate_std = timestamp_data['PseudorangeRateMetersPerSecond'].std()

            # 返回特征字典
            return {
                'num_satellites': num_satellites,
                'gps_count': gps_count,
                'bds_count': bds_count,
                'gps_ratio': gps_ratio,
                'cn0_mean': cn0_mean,
                'cn0_std': cn0_std,
                'cn0_min': cn0_min,
                'cn0_max': cn0_max,
                'elevation_mean': elevation_mean,
                'elevation_std': elevation_std,
                'elevation_min': elevation_min,
                'low_elevation_count': low_elevation_count,
                'high_elevation_count': high_elevation_count,
                'sin_azimuth': sin_azimuth,
                'cos_azimuth': cos_azimuth,
                'azimuth_std': azimuth_std,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'residual_max': residual_max,
                'iono_delay_mean': iono_delay_mean,
                'iono_delay_std': iono_delay_std,
                'tropo_delay_mean': tropo_delay_mean,
                'tropo_delay_std': tropo_delay_std,
                'adr_valid_ratio': adr_valid_ratio,
                'pr_rate_std': pr_rate_std
            }
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return None

    def preprocess_features(self):
        """
        特征预处理：标准化、处理异常值等
        """
        logger.info("预处理特征...")

        # 剔除极端异常值
        self._remove_outliers()

        # 选择使用的特征列
        feature_columns = [
            'num_satellites', 'gps_count', 'bds_count', 'gps_ratio',
            'cn0_mean', 'cn0_std', 'cn0_min',
            'elevation_mean', 'elevation_std', 'low_elevation_count',
            'sin_azimuth', 'cos_azimuth', 'azimuth_std',
            'residual_mean', 'residual_std', 'residual_max',
            'iono_delay_mean', 'tropo_delay_mean', 'adr_valid_ratio'
        ]

        # 确保所有特征列存在
        available_columns = [col for col in feature_columns if col in self.features_df.columns]

        # 提取要使用的特征
        self.features = self.features_df[available_columns].values

        # 特征缩放
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.features = self.scaler.fit_transform(self.features)

        logger.info(f"预处理完成，最终特征形状: {self.features.shape}")
        return self.features

    def _remove_outliers(self, std_threshold=3):
        """
        移除异常值
        """
        # 计算每个特征的均值和标准差
        means = self.features_df.mean()
        stds = self.features_df.std()

        # 初始化掩码
        mask = np.ones(len(self.features_df), dtype=bool)

        # 对每个特征，标记偏离均值超过n个标准差的样本
        for col in self.features_df.columns:
            if self.features_df[col].dtype in [np.float64, np.int64]:
                col_mask = np.abs(self.features_df[col] - means[col]) <= std_threshold * stds[col]
                mask = mask & col_mask

        # 应用掩码
        outliers_count = len(self.features_df) - mask.sum()
        if outliers_count > 0:
            logger.info(f"移除了 {outliers_count} 个异常值 ({outliers_count / len(self.features_df) * 100:.2f}%)")
            self.features_df = self.features_df[mask]
            self.locations = self.locations[mask]
            self.timestamps = self.timestamps[mask]

    def find_optimal_eps_for_dbscan(self, k=4):
        """
        通过k距离图确定DBSCAN的最佳eps参数
        """
        logger.info(f"计算DBSCAN的最佳eps参数 (k={k})...")

        # 计算k距离
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.features)
        distances, _ = neigh.kneighbors(self.features)
        k_dist = np.sort(distances[:, k - 1])

        # 计算k距离的梯度
        gradients = np.gradient(k_dist)

        # 寻找k距离曲线的"肘点"
        knee_point_idx = np.argmax(gradients)
        eps = k_dist[knee_point_idx]

        # 绘制k距离图
        plt.figure(figsize=(10, 6))
        plt.plot(k_dist)
        plt.axhline(y=eps, color='r', linestyle='--')
        plt.title(f'K-Distance Graph (k={k})')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.annotate(f'Suggested eps: {eps:.4f}',
                     xy=(knee_point_idx, eps),
                     xytext=(knee_point_idx + len(k_dist) // 10, eps + 0.5),
                     arrowprops=dict(arrowstyle='->'))
        plt.savefig(self.save_path / f'k_distance_plot_k{k}.png')
        plt.close()

        logger.info(f"建议的eps值: {eps:.4f}")
        return eps

    def run_dbscan(self, eps=None, min_samples=5):
        """
        运行DBSCAN聚类 (CPU版本)
        """
        if eps is None:
            eps = self.find_optimal_eps_for_dbscan()

        logger.info(f"运行DBSCAN聚类 (eps={eps}, min_samples={min_samples})...")

        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.cluster_labels = dbscan.fit_predict(self.features)

        # 统计聚类结果
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)

        logger.info(f"DBSCAN聚类结果: {n_clusters} 个簇, {n_noise} 个噪声点")

        # 计算评估指标
        if n_clusters > 1:
            # 排除噪声点计算评估指标
            mask = self.cluster_labels != -1
            if np.sum(mask) > n_clusters:  # 确保有足够的非噪声点
                sil_score = silhouette_score(self.features[mask], self.cluster_labels[mask])
                db_score = davies_bouldin_score(self.features[mask], self.cluster_labels[mask])
                ch_score = calinski_harabasz_score(self.features[mask], self.cluster_labels[mask])

                logger.info(f"轮廓系数: {sil_score:.4f}")
                logger.info(f"Davies-Bouldin指数: {db_score:.4f}")
                logger.info(f"Calinski-Harabasz指数: {ch_score:.4f}")

        self.dbscan_labels = self.cluster_labels.copy()

        return self.cluster_labels, dbscan

    def run_dbscan_gpu(self, eps=None, min_samples=5):
        """
        内存高效的GPU加速DBSCAN实现 - 仅存储邻居关系而非完整距离矩阵
        """
        try:
            import torch
            from sklearn.cluster import DBSCAN
            from scipy.sparse import csr_matrix

            if not torch.cuda.is_available():
                logger.warning("无可用的CUDA设备，将使用CPU版本的DBSCAN")
                return self.run_dbscan(eps, min_samples)

            if eps is None:
                eps = self.find_optimal_eps_for_dbscan()

            logger.info(f"运行内存优化的GPU加速DBSCAN (eps={eps}, min_samples={min_samples})...")

            # 获取数据尺寸
            n_samples = self.features.shape[0]
            features_tensor = torch.tensor(self.features, device='cuda', dtype=torch.float32)

            # 存储邻居关系，而非完整距离矩阵
            rows = []
            cols = []

            # 分批处理以避免内存溢出
            batch_size = 512  # 可以根据GPU内存调整

            with torch.no_grad():  # 禁用梯度计算以节省内存
                for i in range(0, n_samples, batch_size):
                    end_i = min(i + batch_size, n_samples)
                    batch_i = features_tensor[i:end_i]

                    # 分批计算与其他点的距离
                    for j in range(0, n_samples, batch_size):
                        end_j = min(j + batch_size, n_samples)
                        batch_j = features_tensor[j:end_j]

                        # 计算当前批次的距离矩阵
                        batch_distances = torch.cdist(batch_i, batch_j)

                        # 找出小于eps的距离对应的点对（邻居）
                        neighbor_indices = torch.where(batch_distances <= eps)

                        # 调整索引以对应全局位置
                        global_i = neighbor_indices[0].cpu().numpy() + i
                        global_j = neighbor_indices[1].cpu().numpy() + j

                        # 存储行列索引，用于构建稀疏矩阵
                        rows.extend(global_i.tolist())
                        cols.extend(global_j.tolist())

                    # 释放GPU内存
                    torch.cuda.empty_cache()

            # 构建稀疏邻接矩阵
            adjacency = csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(n_samples, n_samples)
            )

            # 使用DBSCAN处理邻接矩阵
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            self.cluster_labels = dbscan.fit_predict(adjacency)

            # 统计聚类结果
            unique_labels = np.unique(self.cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)

            logger.info(f"DBSCAN聚类结果: {n_clusters} 个簇, {n_noise} 个噪声点")

            # 计算评估指标
            if n_clusters > 1:
                # 排除噪声点计算评估指标
                mask = self.cluster_labels != -1
                if np.sum(mask) > n_clusters:
                    sil_score = silhouette_score(self.features[mask], self.cluster_labels[mask])
                    db_score = davies_bouldin_score(self.features[mask], self.cluster_labels[mask])
                    ch_score = calinski_harabasz_score(self.features[mask], self.cluster_labels[mask])

                    logger.info(f"轮廓系数: {sil_score:.4f}")
                    logger.info(f"Davies-Bouldin指数: {db_score:.4f}")
                    logger.info(f"Calinski-Harabasz指数: {ch_score:.4f}")

            self.dbscan_labels = self.cluster_labels.copy()

            return self.cluster_labels, dbscan

        except Exception as e:
            logger.warning(f"GPU加速DBSCAN失败: {e}，使用CPU版本")
            logger.warning(traceback.format_exc())
            return self.run_dbscan(eps, min_samples)

    def run_kmeans(self, n_clusters_range=range(2, 11)):
        """
        运行K-Means聚类并确定最佳簇数 (CPU版本)
        """
        logger.info(f"寻找K-Means最佳簇数...")

        sil_scores = []
        db_scores = []
        ch_scores = []
        models = {}

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.features)
            models[n_clusters] = (kmeans, cluster_labels)

            # 计算评估指标
            sil = silhouette_score(self.features, cluster_labels)
            db = davies_bouldin_score(self.features, cluster_labels)
            ch = calinski_harabasz_score(self.features, cluster_labels)

            sil_scores.append(sil)
            db_scores.append(db)
            ch_scores.append(ch)

            logger.info(
                f"n_clusters={n_clusters}: 轮廓系数={sil:.4f}, Davies-Bouldin={db:.4f}, Calinski-Harabasz={ch:.4f}")

        # 绘制评估指标图
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(n_clusters_range, sil_scores, 'o-')
        plt.xlabel('簇数量')
        plt.ylabel('轮廓系数 (越高越好)')
        plt.title('轮廓系数评估')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(n_clusters_range, db_scores, 'o-')
        plt.xlabel('簇数量')
        plt.ylabel('Davies-Bouldin指数 (越低越好)')
        plt.title('Davies-Bouldin指数评估')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(n_clusters_range, ch_scores, 'o-')
        plt.xlabel('簇数量')
        plt.ylabel('Calinski-Harabasz指数 (越高越好)')
        plt.title('Calinski-Harabasz指数评估')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_path / 'kmeans_evaluation.png')
        plt.close()

        # 确定最佳簇数
        # 根据轮廓系数
        best_n_sil = n_clusters_range[np.argmax(sil_scores)]
        # 根据Davies-Bouldin指数
        best_n_db = n_clusters_range[np.argmin(db_scores)]

        logger.info(f"根据轮廓系数的最佳簇数: {best_n_sil}")
        logger.info(f"根据Davies-Bouldin指数的最佳簇数: {best_n_db}")

        # 选择最终的最佳簇数 (优先考虑轮廓系数)
        best_n = best_n_sil
        best_model, best_labels = models[best_n]

        self.cluster_labels = best_labels
        self.best_model = best_model
        self.kmeans_labels = self.cluster_labels.copy()

        return best_n, best_model, self.cluster_labels

    def run_kmeans_gpu(self, n_clusters_range=range(2, 11)):
        """
        使用PyTorch实现Windows兼容的GPU加速K-Means聚类
        """
        try:
            import torch
            import torch.nn.functional as F

            if not torch.cuda.is_available():
                logger.warning("无可用的CUDA设备，将使用CPU版本的K-Means")
                return self.run_kmeans(n_clusters_range)

            logger.info(f"寻找GPU加速的K-Means最佳簇数...")

            sil_scores = []
            db_scores = []
            ch_scores = []
            models = {}

            # 将数据转移到GPU
            features_tensor = torch.tensor(self.features, device='cuda', dtype=torch.float32)

            for n_clusters in n_clusters_range:
                # 随机初始化聚类中心
                torch.manual_seed(42)  # 确保可重复性

                # 在数据点中随机选择中心点
                idx = torch.randperm(features_tensor.shape[0], device='cuda')[:n_clusters]
                centroids = features_tensor[idx].clone()

                prev_centroids = torch.zeros_like(centroids)
                labels = torch.zeros(features_tensor.shape[0], dtype=torch.long, device='cuda')

                # 迭代直到收敛或达到最大迭代次数
                max_iter = 100
                tol = 1e-4

                for iteration in range(max_iter):
                    # 计算每个点到每个中心的距离
                    distances = torch.cdist(features_tensor, centroids)

                    # 分配每个点到最近的中心
                    labels = torch.argmin(distances, dim=1)

                    # 更新中心点
                    for k in range(n_clusters):
                        mask = (labels == k)
                        if mask.sum() > 0:
                            centroids[k] = features_tensor[mask].mean(dim=0)

                    # 检查收敛
                    center_shift = torch.norm(centroids - prev_centroids)
                    if center_shift < tol:
                        logger.info(f"K-Means收敛于迭代 {iteration + 1}/{max_iter}")
                        break

                    prev_centroids = centroids.clone()

                    if iteration == max_iter - 1:
                        logger.info(f"K-Means达到最大迭代次数 {max_iter}")

                # 将结果转移到CPU
                cluster_labels = labels.cpu().numpy()
                centroids_cpu = centroids.cpu().numpy()

                # 创建一个伪KMeans模型
                class PyTorchKMeans:
                    def __init__(self, cluster_centers_):
                        self.cluster_centers_ = cluster_centers_

                    def predict(self, X):
                        # 如果GPU可用，使用GPU计算
                        if torch.cuda.is_available():
                            X_tensor = torch.tensor(X, device='cuda', dtype=torch.float32)
                            centers_tensor = torch.tensor(self.cluster_centers_, device='cuda', dtype=torch.float32)
                            distances = torch.cdist(X_tensor, centers_tensor)
                            return torch.argmin(distances, dim=1).cpu().numpy()
                        else:
                            # 回退到CPU实现
                            from sklearn.metrics.pairwise import euclidean_distances
                            distances = euclidean_distances(X, self.cluster_centers_)
                            return np.argmin(distances, axis=1)

                kmeans_model = PyTorchKMeans(centroids_cpu)
                models[n_clusters] = (kmeans_model, cluster_labels)

                # 计算评估指标
                sil = silhouette_score(self.features, cluster_labels)
                db = davies_bouldin_score(self.features, cluster_labels)
                ch = calinski_harabasz_score(self.features, cluster_labels)

                sil_scores.append(sil)
                db_scores.append(db)
                ch_scores.append(ch)

                logger.info(f"n_clusters={n_clusters}: 轮廓系数={sil:.4f}, "
                            f"Davies-Bouldin={db:.4f}, Calinski-Harabasz={ch:.4f}")

            # 绘制评估指标图
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.plot(n_clusters_range, sil_scores, 'o-')
            plt.xlabel('簇数量')
            plt.ylabel('轮廓系数 (越高越好)')
            plt.title('轮廓系数评估')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(n_clusters_range, db_scores, 'o-')
            plt.xlabel('簇数量')
            plt.ylabel('Davies-Bouldin指数 (越低越好)')
            plt.title('Davies-Bouldin指数评估')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(n_clusters_range, ch_scores, 'o-')
            plt.xlabel('簇数量')
            plt.ylabel('Calinski-Harabasz指数 (越高越好)')
            plt.title('Calinski-Harabasz指数评估')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(self.save_path / 'kmeans_gpu_evaluation.png')
            plt.close()

            # 确定最佳簇数
            best_n_sil = n_clusters_range[np.argmax(sil_scores)]
            best_n_db = n_clusters_range[np.argmin(db_scores)]

            logger.info(f"根据轮廓系数的最佳簇数: {best_n_sil}")
            logger.info(f"根据Davies-Bouldin指数的最佳簇数: {best_n_db}")

            # 选择最终的最佳簇数 (优先考虑轮廓系数)
            best_n = best_n_sil
            best_model, best_labels = models[best_n]

            self.cluster_labels = best_labels
            self.best_model = best_model
            self.kmeans_labels = self.cluster_labels.copy()

            return best_n, best_model, self.cluster_labels

        except Exception as e:
            logger.warning(f"GPU加速K-Means失败: {e}，使用CPU版本")
            logger.warning(traceback.format_exc())
            return self.run_kmeans(n_clusters_range)

    def visualize_clusters_2d(self, method='pca'):
        """
        可视化聚类结果 (2D降维)
        """
        if self.cluster_labels is None:
            logger.error("还未进行聚类，无法可视化!")
            return

        logger.info(f"使用 {method} 可视化聚类结果...")

        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2)
            title = 'PCA降维可视化'
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            title = 't-SNE降维可视化'

        reduced_features = reducer.fit_transform(self.features)

        # 可视化
        plt.figure(figsize=(12, 10))

        # 获取唯一的簇标签
        unique_labels = np.unique(self.cluster_labels)

        # 为每个簇分配颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # 绘制每个簇
        for i, label in enumerate(unique_labels):
            if label == -1:
                # 噪声点为黑色
                color = 'k'
                marker = 'x'
                label_name = '噪声'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'簇 {label}'

            mask = self.cluster_labels == label
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                s=50,
                c=[color],
                marker=marker,
                alpha=0.7,
                label=f'{label_name} ({np.sum(mask)} 点)'
            )

        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        plt.savefig(self.save_path / f'cluster_visualization_{method}.png', dpi=300)
        plt.close()

    def analyze_cluster_features(self):
        """
        分析各个簇的特征分布
        """
        if self.cluster_labels is None or self.features_df is None:
            logger.error("请先进行聚类并确保特征数据可用!")
            return

        logger.info("分析各簇的特征特性...")

        # 将聚类标签添加到特征DataFrame
        cluster_df = self.features_df.copy()
        cluster_df['cluster'] = self.cluster_labels

        # 为每个簇计算特征的统计量
        cluster_stats = []
        unique_labels = np.unique(self.cluster_labels)

        for label in unique_labels:
            if label == -1:
                cluster_name = "噪声"
            else:
                cluster_name = f"簇 {label}"

            # 该簇的数据
            cluster_data = cluster_df[cluster_df['cluster'] == label]

            # 计算统计量
            stats = {
                'cluster': cluster_name,
                'count': len(cluster_data),
                'percent': len(cluster_data) / len(cluster_df) * 100
            }

            # 为每个特征计算均值和标准差
            for col in self.features_df.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()
                stats[f'{col}_std'] = cluster_data[col].std()

            cluster_stats.append(stats)

        # 转换为DataFrame
        stats_df = pd.DataFrame(cluster_stats)

        # 保存统计结果
        stats_df.to_csv(self.save_path / 'cluster_statistics.csv', index=False)

        # 为关键特征创建箱线图
        key_features = ['num_satellites', 'cn0_mean', 'elevation_mean', 'residual_mean']

        for feature in key_features:
            if feature not in self.features_df.columns:
                continue

            plt.figure(figsize=(12, 6))

            # 为每个簇创建箱线图
            cluster_data = []
            cluster_names = []

            for label in unique_labels:
                if label == -1:
                    cluster_name = "噪声"
                else:
                    cluster_name = f"簇 {label}"

                # 提取该簇的特征数据
                data = cluster_df[cluster_df['cluster'] == label][feature]

                if len(data) > 0:
                    cluster_data.append(data)
                    cluster_names.append(f"{cluster_name} (n={len(data)})")

            # 创建箱线图
            plt.boxplot(cluster_data, labels=cluster_names)
            plt.title(f'{feature} 在各簇中的分布')
            plt.ylabel(feature)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # 保存图像
            plt.savefig(self.save_path / f'cluster_boxplot_{feature}.png')
            plt.close()

        logger.info(f"簇统计分析已保存到 {self.save_path / 'cluster_statistics.csv'}")

    def visualize_spatial_distribution(self):
        """
        可视化簇的空间分布
        """
        if self.cluster_labels is None or self.locations is None:
            logger.error("请先进行聚类并确保位置数据可用!")
            return

        logger.info("可视化簇的空间分布...")

        # 创建包含位置信息和簇标签的DataFrame
        spatial_df = self.locations.copy()
        spatial_df['cluster'] = self.cluster_labels

        # 绘制经纬度散点图
        plt.figure(figsize=(12, 10))

        # 获取唯一的簇标签
        unique_labels = np.unique(self.cluster_labels)

        # 为每个簇分配颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # 绘制每个簇
        for i, label in enumerate(unique_labels):
            if label == -1:
                # 噪声点为黑色
                color = 'k'
                marker = 'x'
                label_name = '噪声'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'簇 {label}'

            mask = self.cluster_labels == label
            plt.scatter(
                spatial_df[mask]['lon'],
                spatial_df[mask]['lat'],
                s=50,
                c=[color],
                marker=marker,
                alpha=0.7,
                label=f'{label_name} ({np.sum(mask)} 点)'
            )

        plt.title('簇的空间分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        plt.savefig(self.save_path / 'spatial_distribution.png', dpi=300)
        plt.close()

    def save_model(self, filename='gnss_cluster_model.pkl'):
        """
        保存聚类模型和相关信息
        """
        if self.best_model is None:
            logger.warning("没有最佳模型可保存!")
            return

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'labels': self.cluster_labels,
            'scaler_type': self.scaler_type
        }

        joblib.dump(model_data, self.save_path / filename)
        logger.info(f"模型已保存到 {self.save_path / filename}")

    def load_model(self, filename='gnss_cluster_model.pkl'):
        """
        加载保存的聚类模型
        """
        model_path = self.save_path / filename
        if not model_path.exists():
            logger.error(f"模型文件 {model_path} 不存在!")
            return False

        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.cluster_labels = model_data['labels']
            self.scaler_type = model_data.get('scaler_type', 'robust')

            logger.info(f"成功加载模型 {model_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            return False

    def save_labeled_data(self, dbscan_labels=None, kmeans_labels=None, output_dir=None):
        """
        将DBSCAN和K-Means聚类标签同时与原始数据关联并保存为CSV文件
        """
        # 使用当前模型标签或传入的标签
        if dbscan_labels is None and hasattr(self, 'dbscan_labels'):
            dbscan_labels = self.dbscan_labels
        elif dbscan_labels is None and self.cluster_labels is not None:
            dbscan_labels = self.cluster_labels

        if kmeans_labels is None and hasattr(self, 'kmeans_labels'):
            kmeans_labels = self.kmeans_labels

        if dbscan_labels is None and kmeans_labels is None:
            logger.error("没有可用的聚类标签!")
            return False

        if output_dir is None:
            output_dir = self.save_path
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建包含原始时间戳、位置和聚类标签的DataFrame
        labeled_data = pd.DataFrame({
            'utcTimeMillis': self.timestamps,
            'lat': self.locations['lat'],
            'lon': self.locations['lon'],
            'alt': self.locations['alt'],
            'wls_x': self.locations['wls_x'],
            'wls_y': self.locations['wls_y'],
            'wls_z': self.locations['wls_z']
        })

        # 添加DBSCAN标签
        if dbscan_labels is not None:
            labeled_data['dbscan_label'] = dbscan_labels

        # 添加K-Means标签
        if kmeans_labels is not None:
            labeled_data['kmeans_label'] = kmeans_labels

        # 添加对所用卫星系统的记录
        labeled_data['constellations_used'] = 'GPS_L1_CA,BDS_B1_I'

        # 保存为CSV
        labeled_data.to_csv(output_dir / 'clustered_data.csv', index=False)

        # 创建映射字典: timestamp -> cluster_labels
        timestamp_to_dbscan = {}
        timestamp_to_kmeans = {}

        if dbscan_labels is not None:
            timestamp_to_dbscan = dict(zip(self.timestamps, dbscan_labels))
        if kmeans_labels is not None:
            timestamp_to_kmeans = dict(zip(self.timestamps, kmeans_labels))

        # 遍历原始数据路径
        processed_files = 0

        # 递归查找所有ground_truth.csv文件
        for ground_truth_file in Path(self.data_path).glob('**/ground_truth.csv'):
            try:
                # 读取原始ground_truth文件
                gt_df = pd.read_csv(ground_truth_file)

                # 为每个时间戳添加聚类标签
                if dbscan_labels is not None:
                    gt_df['dbscan_label'] = gt_df['utcTimeMillis'].map(
                        timestamp_to_dbscan).fillna(-99)  # -99表示未聚类的点

                if kmeans_labels is not None:
                    gt_df['kmeans_label'] = gt_df['utcTimeMillis'].map(
                        timestamp_to_kmeans).fillna(-99)  # -99表示未聚类的点

                # 保存带有标签的新文件 - 保持原目录结构
                rel_path = ground_truth_file.relative_to(self.data_path)
                output_file = output_dir / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                gt_df.to_csv(output_file, index=False)

                processed_files += 1

            except Exception as e:
                logger.error(f"处理文件 {ground_truth_file} 时出错: {e}")

        logger.info(f"成功处理 {processed_files} 个文件，DBSCAN和K-Means标签已保存到 {output_dir}")
        return True


# 主程序
if __name__ == "__main__":
    # 设置路径
    data_path = Path(r'G:\毕业论文\data\processed_data')
    save_path = Path(r'G:\毕业论文\results\by_clustering')

    # 验证路径存在
    if not data_path.exists():
        logger.error(f"数据路径不存在: {data_path}")
        exit(1)

    # 创建保存目录
    save_path.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    model = GNSSClusteringModel(data_path, save_path, scaler_type='robust')

    # 使用并行模式加载数据
    if model.load_data():
        # 预处理特征
        model.preprocess_features()

        # 运行DBSCAN聚类
        logger.info("=== 运行DBSCAN聚类 ===")
        if model.acceleration['gpu']:
            dbscan_labels, dbscan_model = model.run_dbscan_gpu(min_samples=3)
        else:
            dbscan_labels, dbscan_model = model.run_dbscan(min_samples=3)

        # 可视化DBSCAN结果
        model.visualize_clusters_2d(method='pca')
        model.visualize_clusters_2d(method='tsne')
        model.analyze_cluster_features()
        model.visualize_spatial_distribution()

        # 运行K-Means聚类
        logger.info("=== 运行K-Means聚类 ===")
        if model.acceleration['gpu']:
            best_k, kmeans_model, kmeans_labels = model.run_kmeans_gpu()
        else:
            best_k, kmeans_model, kmeans_labels = model.run_kmeans()

        # 可视化K-Means结果
        model.visualize_clusters_2d(method='pca')
        model.visualize_clusters_2d(method='tsne')
        model.analyze_cluster_features()
        model.visualize_spatial_distribution()

        # 保存带有两种聚类标签的数据
        model.save_labeled_data(
            dbscan_labels=model.dbscan_labels,
            kmeans_labels=model.kmeans_labels,
            output_dir=Path(r'G:\毕业论文\data\labeled_data')
        )

        # 保存模型
        model.save_model('dbscan_model.pkl')
        model.save_model('kmeans_model.pkl')

        logger.info("聚类分析完成!")
    else:
        logger.error("数据加载失败!")