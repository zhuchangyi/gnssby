import pandas as pd
import os
from pathlib import Path

# 假设 ground_truth_df 是你的 ground truth DataFrame
# 确定哪些列名重复,这些是你想要删除的列
current_script_path = Path(__file__).resolve()
root_path = current_script_path.parents[2]

# 构建到 'data/processed' 和 'data/raw' 的路径
filtered_path = root_path / "data" / "processed"

for subdir, dirs, files in os.walk(filtered_path):

    ground_truth_file = os.path.join(subdir, 'ground_truth.csv')
    if os.path.exists(ground_truth_file):
        # 读取 ground_truth.csv 文件
        ground_truth_df = pd.read_csv(ground_truth_file)

        # 获取需要删除的重复列名
        duplicated_columns = [col for col in ground_truth_df.columns if '.1' in col or '.2' in col]

        # 删除重复的列
        ground_truth_df = ground_truth_df.drop(columns=duplicated_columns)

        # 将处理后的 DataFrame 保存到原处
        ground_truth_df.to_csv(ground_truth_file, index=False)
