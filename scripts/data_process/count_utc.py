import pandas as pd
import logging
import os
import numpy as np


# Set up logging
# logging.basicConfig(filename='timestamp_count.log', level=logging.INFO,
#                     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
#
# for subdir, dir, files in os.walk(r'../kaggle2023/filtered_data'):
#     gnss_file = os.path.join(subdir, 'gnss_data.csv')
#     if os.path.exists(gnss_file):
#         gnss_df = pd.read_csv(gnss_file)
#         unique_timestamps = gnss_df['utcTimeMillis'].unique()
#         logging.info(f'The number of unique timestamps is: {len(unique_timestamps)}')

# with open('timestamp_count.log', 'r') as f:
#     lines = f.readlines()
#     num_timestamps = 0
#     for line in lines:
#         num_timestamps += int(line.split(' ')[-1])
#     print(f'The total number of unique timestamps is: {num_timestamps}')
#     # The total number of unique timestamps is: 251671


def _correction_to_one_hot(correction):
    vector = np.zeros(20)
    # 四舍五入修正值
    correction_rounded = round(correction)
    # 将修正值限制在-10到10之间，这一步确保不超出向量表示范围
    correction_clipped = np.clip(correction_rounded, -10, 10)
    # 计算修正值对应的向量索引
    if correction_clipped >= 0:
        index = int(correction_clipped + 9)  # 对于非负修正，0对应向量中的第10位
    else:
        index = int(correction_clipped + 10)  # 对于负修正，-1对应向量中的第9位
    vector[index] = 1
    return vector


print(_correction_to_one_hot(65.5))


def error_to_one_hot_with_zero(error):
    # 限制误差范围，并考虑-0.5到0.5四舍五入到0的情况
    error = np.clip(np.round(error), -10, 10)
    one_hot_vector = np.zeros(21)
    index = int(error + 10)
    one_hot_vector[index] = 1
    return one_hot_vector


# 重新测试不同的误差值，包括-0.5到0.5之间的值
test_errors_with_zero = [5.6, 5.3, -5.6, -6.3, 15, -20, -0.5, 0.3,0,10]
test_results_with_zero = np.array([error_to_one_hot_with_zero(error) for error in test_errors_with_zero])



print(test_results_with_zero)


