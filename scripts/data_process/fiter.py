import pandas as pd
import os
import multiprocessing
import numpy as np
import shutil
from pathlib import Path
ADR_STATE_UNKNOWN = 0
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2
ADR_STATE_HALF_CYCLE_RESOLVED = 1 << 3
ADR_STATE_HALF_CYCLE_REPORTED = 1 << 4
num={}
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
def filter_and_keep_top_20(df):
    # 检查数据是否满足筛选条件
    condition =(df['SvElevationDegrees'] >= 10)
    if condition.all():
        # 如果所有数据都满足筛选条件，则保留所有数据
        return df
    else:
        # 如果有数据不满足筛选条件，则只保留前20条数据
        df_sorted = df.sort_values(by=['StateWeight', 'SvElevationDegrees'], ascending=[False, False])
        return df_sorted.head(20)


def filterdata(file_path,gt_path,out_path):
    #print("processing")
    # 加载CSV文件
    gnss_data = pd.read_csv(file_path, low_memory=False)
    gt_data = pd.read_csv(gt_path,low_memory=False)
    gt_timestamps = set(gt_data['utcTimeMillis'])

    # 过滤掉那些不在真值数据时间戳中的观测数据记录
    gnss_data = gnss_data[gnss_data['utcTimeMillis'].isin(gt_timestamps)]

    # 将第21列转换为浮点数，无法转换的设置为NaN
    gnss_data.iloc[:, 21] = pd.to_numeric(gnss_data.iloc[:, 21], errors='coerce')

    # 检查转换后的数据类型统计
    column_types_after = gnss_data.iloc[:, 21].apply(type).value_counts()
    #print(column_types_after)

    # 设置筛选阈值
    elevation_threshold = 10  # 仰角阈值，单位为度
    drift_threshold = 5  # 载波频率漂移阈值，单位为米/秒

    gnss_data['StateWeight'] = gnss_data['AccumulatedDeltaRangeState'].apply(get_state_weight)
    # gnss_data = gnss_data[gnss_data['SvElevationDegrees'] >= elevation_threshold]

    # 对每个时间戳的数据应用上述函数
    gnss_data = gnss_data.groupby('utcTimeMillis').apply(filter_and_keep_top_20)

    # 重置索引
    gnss_data = gnss_data.reset_index(drop=True)

    # 按照状态权重排序数据
    sorted_data = gnss_data.sort_values(by=['utcTimeMillis', 'StateWeight', 'SvElevationDegrees'],
                                        ascending=[True, False, False])
    utc1 = np.array([gnss_data['utcTimeMillis'].unique()])
    utc2 = np.array([gt_data['utcTimeMillis']])
    print(utc1.shape[1] - utc2.shape[1])

    # 计算每个时间戳的观测数量
    observations_per_timestamp = sorted_data.groupby('utcTimeMillis').size()
    num[str(file_path)] = observations_per_timestamp  # 用文件路径作为键保存观测数量



    # # 排序筛选后的数据，首先按照utcTimeMillis升序排列，
    # # 然后再按照SvClockDriftMetersPerSecond和SvElevationDegrees降序排列
    # sorted_data = filtered_data.sort_values(
    #     by=['utcTimeMillis' ,'SvClockDriftMetersPerSecond', 'SvElevationDegrees', ],
    #     ascending=[True, True, False]
    # )
    #
    # # 计算每个时间戳的观测数量
    # observations_per_timestamp = sorted_data.groupby('utcTimeMillis').size()
    # num[str(file_path)]=observations_per_timestamp
    # #num.append(observations_per_timestamp)

    # 保存排序后的数据到一个新的CSV文件
    # 确定CSV文件的完整路径
    out_csv_path = os.path.abspath(os.path.join(out_path, "gnss_data.csv"))
    # 获取CSV文件的父目录路径
    out_dir = os.path.dirname(out_csv_path)
    # 如果父目录不存在，则创建它
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sorted_data.to_csv(out_csv_path, index=False)


    # 输出每个时间戳的观测数量
    #print("Observations per timestamp:")
    #print(observations_per_timestamp)

    #print(f"\nSorted data saved to {output_file_path}")

def process_data(args):
    file_path, gt_path, out_path = args
    filterdata(file_path, gt_path, out_path)

# og_path=os.path.join("../kaggle2023")
# data_path = os.path.join(og_path, r"sdc2023\train")
# filtered_path = os.path.join(og_path,"filtered_data")
#print(data_path)



if __name__ == "__main__":
    current_script_path = Path(__file__).resolve()

    # 项目根目录的路径
    root_path = current_script_path.parents[2]

    # 构建到 'data/processed' 和 'data/raw' 的路径
    filtered_path = root_path / "data" / "processed_data"
    data_raw_path = root_path / "data" / "raw"

    # 如果您需要构建 'data/sdc2023/train' 的路径
    data_path = data_raw_path / "sdc2023" / "train"
    # 创建一个包含所有任务的列表
    tasks = []
    for trace in os.listdir(data_path):
        trace_path = os.path.join(data_path, trace)
        if os.path.isdir(trace_path):  # 确保trace是一个目录
            for phones in os.listdir(trace_path):
                phone_path = os.path.join(trace_path, phones)
                if os.path.isdir(phone_path):  # 确保phones是一个目录
                    out_path = os.path.join(filtered_path, trace, phones)
                    gnss_csv_path = os.path.join(phone_path, "device_gnss.csv")
                    gt_path = os.path.join(phone_path, 'ground_truth.csv')
                    tasks.append((gnss_csv_path, gt_path, out_path))

    # 创建一个进程池
    pool = multiprocessing.Pool()
    pool.map(process_data, tasks)
print(num)




