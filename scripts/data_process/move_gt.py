from pathlib import Path
from src.gnss_lib import coordinates as coord
import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path

current_script_path = Path(__file__).resolve()
root_path = current_script_path.parents[2]
print(current_script_path, root_path)

rawdata_path = root_path / "data" / "raw" / "sdc2023" / "train"
processed_path = root_path / "data" / "processed_data"
log_path = root_path / "logs"
log_file_path = log_path / 'count_after_remove.log'


# 遍历data_raw_path目录下的所有子目录和文件
# for path in data_raw_path.rglob('ground_truth.csv'):
#     # 目标目录路径
#     processed_trace_path = filtered_path / path.relative_to(data_raw_path)
#     processed_trace_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在
#     shutil.copy(path, processed_trace_path)  # 复制文件

#算true correction的代码
for subdir, dirs, files in os.walk(processed_path):
    ground_truth_file = os.path.join(subdir, 'ground_truth.csv')
    if os.path.exists(ground_truth_file):
        gt = pd.read_csv(ground_truth_file)
        gt_geo = gt[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
        gt_ecef = coord.geodetic2ecef(gt_geo)
        gt_ecef_df = pd.DataFrame(gt_ecef, columns=['gt_ecef_X', 'gt_ecef_Y', 'gt_ecef_Z'])

        # 在合并之前删除原始DataFrame中的ECEF列（如果存在）
        for col in ['gt_ecef_X', 'gt_ecef_Y', 'gt_ecef_Z']:
            if col in gt.columns:
                gt.drop(columns=[col], inplace=True)

        # 执行合并
        gt = gt.merge(gt_ecef_df, left_index=True, right_index=True, how='left')

        # 计算真实修正值
        gt['true_correction_x'] = gt['WlsPositionXEcefMeters'] - gt['gt_ecef_X']
        gt['true_correction_y'] = gt['WlsPositionYEcefMeters'] - gt['gt_ecef_Y']
        gt['true_correction_z'] = gt['WlsPositionZEcefMeters'] - gt['gt_ecef_Z']

        # 保存新的DataFrame到CSV
        gt.to_csv(ground_truth_file, index=False)
    else:
        print(f"File does not exist: {ground_truth_file}")