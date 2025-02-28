import pandas as pd
from pyproj import Transformer
import os
import concurrent.futures

base_path = '/Users/park/PycharmProjects/gnss/sdc2023/train/'

# 设定输入文件路径
#gnss_csv_path = '/Users/park/PycharmProjects/gnss/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4/device_gnss.csv'
#ground_truth_csv_path = '/Users/park/PycharmProjects/gnss/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4/ground_truth.csv'


# 初始化坐标转换器：ECEF到WGS84
transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# 定义ECEF到WGS84的转换函数
def ecef_to_wgs84(row):
    lon, lat, alt = transformer.transform(row['WlsPositionXEcefMeters'], row['WlsPositionYEcefMeters'], row['WlsPositionZEcefMeters'])
    return pd.Series({'WlsLatitudeDegrees': lat, 'WlsLongitudeDegrees': lon, 'WlsAltitudeMeters': alt})


def process_trace_phone(trace, phone_model):
    gnss_csv_path = os.path.join(base_path, trace, phone_model, 'device_gnss.csv')
    ground_truth_csv_path = os.path.join(base_path, trace, phone_model, 'ground_truth.csv')
    gnss_df = pd.read_csv(gnss_csv_path)
    ground_truth_df = pd.read_csv(ground_truth_csv_path)

    # 过滤重复的 WLS 数据，只保留唯一的时间戳记录
    wls_df = gnss_df[['utcTimeMillis', 'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters',
                      'WlsPositionZEcefMeters']].drop_duplicates(subset=['utcTimeMillis'])
    ground_truth_df.rename(columns={'UnixTimeMillis': 'utcTimeMillis'}, inplace=True)
    # 应用转换并添加经纬度到 WLS 数据框中
    wls_df[['WlsLatitudeDegrees', 'WlsLongitudeDegrees', 'WlsAltitudeMeters']] = wls_df.apply(ecef_to_wgs84, axis=1)

    merged_df = ground_truth_df.merge(wls_df, on='utcTimeMillis', how='left')
    merged_df.to_csv(ground_truth_csv_path, index=False)
    print(f"Processed trace {trace} with phone {phone_model}")

# process_trace_phone('2023-09-07-22-48-us-ca-routebc2','pixel4xl')
traces = [d.name for d in os.scandir(base_path) if d.is_dir()]

# 对于每个trace，获取其下的所有phone_model
trace_phone_pairs = []
for trace in traces:
    trace_path = os.path.join(base_path, trace)
    phone_models = [d.name for d in os.scandir(trace_path) if d.is_dir()]
    for phone_model in phone_models:
        trace_phone_pairs.append((trace, phone_model))

# 使用 ThreadPoolExecutor 并发处理每个trace和phone_model组合
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 向executor提交任务
    future_to_trace_phone = {executor.submit(process_trace_phone, trace, phone_model): (trace, phone_model) for
                             trace, phone_model in trace_phone_pairs}

    # 等待每个任务完成，并打印结果
    for future in concurrent.futures.as_completed(future_to_trace_phone):
        trace, phone_model = future_to_trace_phone[future]
        try:
            future.result()
        except Exception as exc:
            print(f'Trace {trace} with phone {phone_model} generated an exception: {exc}')



