import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import tqdm
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gnss_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for ADR states
ADR_STATE_UNKNOWN = 0
ADR_STATE_VALID = 1 << 0
ADR_STATE_RESET = 1 << 1
ADR_STATE_CYCLE_SLIP = 1 << 2
ADR_STATE_HALF_CYCLE_RESOLVED = 1 << 3
ADR_STATE_HALF_CYCLE_REPORTED = 1 << 4


def get_state_weight(state):
    """Calculate weight based on ADR state flags."""
    weight = 0
    if state & ADR_STATE_VALID:
        weight += 32  # VALID state has highest weight
    if state & ADR_STATE_HALF_CYCLE_RESOLVED:
        weight += 16  # HALF_CYCLE_RESOLVED is second
    if state & ADR_STATE_HALF_CYCLE_REPORTED:
        weight += 8  # HALF_CYCLE_REPORTED is third
    # Unknown or other states get no additional points
    return weight


def correction_to_one_hot(correction):
    """Convert correction value to one-hot encoded vector."""
    one_hot_vector = np.zeros(12)

    if -25 <= correction < -15:
        index = 0
    elif -15 <= correction < -10:
        index = 1
    elif -10 <= correction < -5:
        index = 2
    elif -5 <= correction < -2:
        index = 3
    elif -2 <= correction < -1:
        index = 4
    elif -1 <= correction < 0:
        index = 5
    elif 0 <= correction <= 1:
        index = 6
    elif 1 < correction <= 2:
        index = 7
    elif 2 < correction <= 5:
        index = 8
    elif 5 < correction <= 10:
        index = 9
    elif 10 < correction <= 15:
        index = 10
    elif 15 < correction <= 25:
        index = 11
    else:
        # Default to middle index if out of range
        index = 6

    one_hot_vector[index] = 1
    return one_hot_vector


def _calculate_distance(x1, y1, z1, x2, y2, z2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def collect_global_statistics(root_dir, exclude_traces):
    """Collect global statistics for normalization."""
    all_velocity_data = []
    all_clock_bias_data = []
    all_pseudorange_rate_data = []
    all_elevation_tan_data = []
    all_cn0_dbhz_data = []
    all_ionospheric_delay_data = []
    all_tropospheric_delay_data = []
    all_azimuth_data = []
    all_pseudorange_residuals = []

    for subdir, dirs, files in os.walk(root_dir):
        norm_subdir = os.path.normpath(subdir)
        should_exclude = any(exclude_trace in norm_subdir for exclude_trace in exclude_traces)
        if should_exclude:
            logger.info(f"Excluding {subdir} from global statistics collection")
            continue

        path_parts = norm_subdir.split(os.path.sep)
        if len(path_parts) < 2:
            continue

        gnss_file = os.path.join(subdir, 'gnss_data.csv')
        ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

        if not (os.path.exists(gnss_file) and os.path.exists(ground_truth_file)):
            continue

        try:
            gnss_df = pd.read_csv(gnss_file)
            ground_truth_df = pd.read_csv(ground_truth_file)

            if ground_truth_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters',
                                'WlsPositionXEcefMeters']].isnull().any().any():
                continue

            columns_to_remove = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
            gnss_df = gnss_df.drop(columns=columns_to_remove, errors='ignore')
            merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')

            for timestamp in merged_df['utcTimeMillis'].unique():
                timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp].copy()
                timestamp_data = timestamp_data[~timestamp_data['SignalType'].isin(['QZS_L1_CA', 'QZS_L5_Q'])]

                # Calculate pseudorange residuals for global scaling
                for index, row in timestamp_data.iterrows():
                    try:
                        distance = _calculate_distance(
                            row['WlsPositionXEcefMeters'], row['WlsPositionYEcefMeters'], row['WlsPositionZEcefMeters'],
                            row['SvPositionXEcefMeters'], row['SvPositionYEcefMeters'], row['SvPositionZEcefMeters']
                        )

                        residual = row['RawPseudorangeMeters'] - distance
                        all_pseudorange_residuals.append(residual)
                    except (KeyError, ValueError) as e:
                        # Skip this row if there's a problem calculating the residual
                        continue

                # Extract feature data for normalization
                try:
                    gnss_features = timestamp_data[
                        ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                         'SvVelocityZEcefMetersPerSecond', 'SvClockBiasMeters', 'PseudorangeRateMetersPerSecond',
                         'SvElevationDegrees', 'Cn0DbHz', 'IonosphericDelayMeters', 'TroposphericDelayMeters',
                         'SvAzimuthDegrees']
                    ].copy()

                    # Replace NaN with column medians
                    for col in gnss_features.columns:
                        gnss_features[col].fillna(gnss_features[col].median(), inplace=True)

                    # Convert to numpy array after filling NaNs
                    gnss_features = gnss_features.to_numpy()

                    # Add to global collections
                    all_velocity_data.append(gnss_features[:, :3])
                    all_clock_bias_data.append(gnss_features[:, 3])
                    all_pseudorange_rate_data.append(gnss_features[:, 4])
                    all_elevation_tan_data.append(np.tan(np.radians(gnss_features[:, 5])))
                    all_cn0_dbhz_data.append(gnss_features[:, 6])
                    all_ionospheric_delay_data.append(gnss_features[:, 7])
                    all_tropospheric_delay_data.append(gnss_features[:, 8])
                    all_azimuth_data.append(gnss_features[:, 9])
                except Exception as e:
                    logger.warning(f"Error processing features in {subdir} at timestamp {timestamp}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error processing directory {subdir}: {e}")
            continue

    # Stack and reshape all collected data
    scalers = {}

    try:
        if all_velocity_data:
            all_velocity_data = np.vstack(all_velocity_data)
            scalers['velocity'] = StandardScaler().fit(all_velocity_data)

        if all_clock_bias_data:
            all_clock_bias_data = np.hstack(all_clock_bias_data).reshape(-1, 1)
            scalers['clock_bias'] = StandardScaler().fit(all_clock_bias_data)

        if all_pseudorange_rate_data:
            all_pseudorange_rate_data = np.hstack(all_pseudorange_rate_data).reshape(-1, 1)
            scalers['pseudorange_rate'] = StandardScaler().fit(all_pseudorange_rate_data)

        if all_elevation_tan_data:
            all_elevation_tan_data = np.hstack(all_elevation_tan_data).reshape(-1, 1)
            scalers['elevation_tan'] = StandardScaler().fit(all_elevation_tan_data)

        if all_cn0_dbhz_data:
            all_cn0_dbhz_data = np.hstack(all_cn0_dbhz_data).reshape(-1, 1)
            scalers['cn0_dbhz'] = StandardScaler().fit(all_cn0_dbhz_data)

        if all_ionospheric_delay_data:
            all_ionospheric_delay_data = np.hstack(all_ionospheric_delay_data).reshape(-1, 1)
            scalers['ionospheric_delay'] = StandardScaler().fit(all_ionospheric_delay_data)

        if all_tropospheric_delay_data:
            all_tropospheric_delay_data = np.hstack(all_tropospheric_delay_data).reshape(-1, 1)
            scalers['tropospheric_delay'] = StandardScaler().fit(all_tropospheric_delay_data)

        if all_azimuth_data:
            all_azimuth_data = np.hstack(all_azimuth_data).reshape(-1, 1)
            scalers['azimuth'] = StandardScaler().fit(all_azimuth_data)

        if all_pseudorange_residuals:
            all_pseudorange_residuals = np.array(all_pseudorange_residuals).reshape(-1, 1)
            scalers['pseudorange_residual'] = MinMaxScaler(feature_range=(0, 1)).fit(all_pseudorange_residuals)

        return scalers

    except Exception as e:
        logger.error(f"Error creating scalers: {e}")
        raise


def process_trace(subdir, exclude_traces, signal_max_satellites, signal_order, scalers):
    """Process a single trace directory."""
    trace_data = {}
    norm_subdir = os.path.normpath(subdir)
    should_exclude = any(exclude_trace in norm_subdir for exclude_trace in exclude_traces)

    if should_exclude:
        logger.info(f"Skipping excluded trace: {subdir}")
        return None

    path_parts = norm_subdir.split(os.path.sep)
    if len(path_parts) < 2:
        return None

    trace_id = path_parts[-2]  # trace ID is the second last part of the path
    device_id = path_parts[-1]  # device ID is the last part of the path
    trace_device_id = f"{trace_id}_{device_id}"

    gnss_file = os.path.join(subdir, 'gnss_data.csv')
    ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

    if not (os.path.exists(gnss_file) and os.path.exists(ground_truth_file)):
        return None

    try:
        gnss_df = pd.read_csv(gnss_file)
        ground_truth_df = pd.read_csv(ground_truth_file)

        if ground_truth_df[
            ['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters', 'WlsPositionXEcefMeters']].isnull().any().any():
            return None

        columns_to_remove = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
        gnss_df = gnss_df.drop(columns=columns_to_remove, errors='ignore')
        merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')

        trace_data[trace_device_id] = {
            'features': [],
            'labels': [],
            'true_corrections': [],
            'wls_ecef': []
        }

        for timestamp in merged_df['utcTimeMillis'].unique():
            timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp].copy()
            timestamp_data = timestamp_data[~timestamp_data['SignalType'].isin(['QZS_L1_CA', 'QZS_L5_Q'])]

            # Skip if no rows left after filtering
            if len(timestamp_data) == 0:
                continue

            wls_ecef_coords = \
            timestamp_data[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].iloc[
                0].to_numpy()
            trace_data[trace_device_id]['wls_ecef'].append(wls_ecef_coords)

            # Calculate ADR state weights and assign
            timestamp_data.loc[:, 'StateWeight'] = timestamp_data['AccumulatedDeltaRangeState'].apply(get_state_weight)

            # Sort by signal order, state weight, and elevation
            timestamp_data['Order'] = timestamp_data['SignalType'].map(signal_order)
            timestamp_data.sort_values(by=['Order', 'StateWeight', 'SvElevationDegrees'],
                                       ascending=[True, False, False], inplace=True)

            # Skip if true_correction columns don't exist
            if not all(col in timestamp_data.columns for col in
                       ['true_correction_x', 'true_correction_y', 'true_correction_z']):
                continue

            true_correction = np.array([
                timestamp_data['true_correction_x'].iloc[0],
                timestamp_data['true_correction_y'].iloc[0],
                timestamp_data['true_correction_z'].iloc[0]
            ])

            # Skip if any correction is out of range
            if np.any(np.abs(true_correction) > 25):
                continue

            pseudorange_residuals = []
            los_vectors = []

            for _, row in timestamp_data.iterrows():
                try:
                    distance = _calculate_distance(
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
                    los_vector_normalized = los_vector / np.linalg.norm(los_vector)
                    los_vectors.append(los_vector_normalized)
                except (KeyError, ValueError, ZeroDivisionError) as e:
                    # Skip this row if any required data is missing or invalid
                    continue

            # Skip if no valid data
            if not los_vectors:
                continue

            pseudorange_residuals = np.array(pseudorange_residuals)
            los_vectors = np.stack(los_vectors)

            # Extract features
            features_columns = [
                'Cn0DbHz', 'IonosphericDelayMeters', 'TroposphericDelayMeters',
                'SvElevationDegrees', 'SvAzimuthDegrees',
                'PseudorangeRateMetersPerSecond', 'PseudorangeRateUncertaintyMetersPerSecond',
                'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                'SvVelocityZEcefMetersPerSecond', 'SvClockBiasMeters'
            ]

            # Calculate median values for each column to use for NaN filling
            median_values = timestamp_data[features_columns].median()

            # Fill NaN values with column-wise medians
            gnss_features = timestamp_data[features_columns].fillna(median_values).to_numpy()

            # Convert elevation angle to tan and standardize
            gnss_features[:, 3] = np.tan(np.radians(gnss_features[:, 3]))
            gnss_features[:, 3] = scalers['elevation_tan'].transform(gnss_features[:, 3].reshape(-1, 1)).flatten()

            # Apply global scalers to each feature
            gnss_features[:, 7:10] = scalers['velocity'].transform(gnss_features[:, 7:10])
            gnss_features[:, 10] = scalers['clock_bias'].transform(gnss_features[:, 10].reshape(-1, 1)).flatten()
            gnss_features[:, 5] = scalers['pseudorange_rate'].transform(gnss_features[:, 5].reshape(-1, 1)).flatten()
            gnss_features[:, 0] = scalers['cn0_dbhz'].transform(gnss_features[:, 0].reshape(-1, 1)).flatten()
            gnss_features[:, 1] = scalers['ionospheric_delay'].transform(gnss_features[:, 1].reshape(-1, 1)).flatten()
            gnss_features[:, 2] = scalers['tropospheric_delay'].transform(gnss_features[:, 2].reshape(-1, 1)).flatten()
            gnss_features[:, 4] = scalers['azimuth'].transform(gnss_features[:, 4].reshape(-1, 1)).flatten()

            # Apply pre-fitted scaler to pseudorange residuals
            pseudorange_residuals = scalers['pseudorange_residual'].transform(
                pseudorange_residuals.reshape(-1, 1)).flatten()

            # Combine features with line-of-sight vectors and pseudorange residuals
            gnss_features = np.hstack([gnss_features, los_vectors, pseudorange_residuals[:, np.newaxis]])

            # Process each signal type according to max satellites configuration
            combined_features = None
            for signal_type in signal_max_satellites:
                type_mask = (timestamp_data['SignalType'] == signal_type)
                type_features = gnss_features[type_mask]

                # Pad or truncate to configured size
                max_count = signal_max_satellites[signal_type]
                current_count = type_features.shape[0]

                if current_count < max_count:
                    # Pad with zeros if fewer satellites than max
                    padding_size = max_count - current_count
                    padding = np.zeros((padding_size, gnss_features.shape[1]))
                    type_features = np.vstack([type_features, padding])
                elif current_count > max_count:
                    # Keep only the configured maximum number
                    type_features = type_features[:max_count]

                if combined_features is None:
                    combined_features = type_features
                else:
                    combined_features = np.vstack([combined_features, type_features])

            # Convert corrections to one-hot encoding
            correction_x = correction_to_one_hot(timestamp_data['true_correction_x'].iloc[0])
            correction_y = correction_to_one_hot(timestamp_data['true_correction_y'].iloc[0])
            correction_z = correction_to_one_hot(timestamp_data['true_correction_z'].iloc[0])

            # Store processed data
            trace_data[trace_device_id]['features'].append(combined_features)
            trace_data[trace_device_id]['labels'].append(
                np.stack([correction_x, correction_y, correction_z], axis=-1))
            trace_data[trace_device_id]['true_corrections'].append(true_correction)

        return trace_data

    except Exception as e:
        logger.error(f"Error processing trace {subdir}: {e}")
        return None


def find_all_subdirs(root_dir):
    """Find all subdirectories to process."""
    subdirs = []
    for subdir, dirs, files in os.walk(root_dir):
        if 'gnss_data.csv' in files and 'ground_truth.csv' in files:
            subdirs.append(subdir)
    return subdirs


def save_trace_data(trace_data, save_dir):
    """Save processed trace data to files."""
    for trace_device_id, data in trace_data.items():
        if data['features'] and data['labels'] and data['true_corrections'] and data['wls_ecef']:
            np.save(save_dir / f'features_{trace_device_id}.npy', np.array(data['features']))
            np.save(save_dir / f'labels_{trace_device_id}.npy', np.array(data['labels']))
            np.save(save_dir / f'true_corrections_{trace_device_id}.npy', np.array(data['true_corrections']))
            np.save(save_dir / f'wls_ecef_{trace_device_id}.npy', np.array(data['wls_ecef']))
            logger.info(f"Saved data for {trace_device_id}")


def save_scalers(scalers, scaler_path):
    """Save all scalers to disk."""
    os.makedirs(scaler_path, exist_ok=True)
    for name, scaler in scalers.items():
        joblib.dump(scaler, os.path.join(scaler_path, f'scaler_{name}.pkl'))
    logger.info(f"Saved all scalers to {scaler_path}")


def preprocess_data(root_dir, save_dir, scaler_path, num_workers=None):
    """
    Preprocess GNSS data for scene recognition using parallel processing.

    Args:
        root_dir: Path to raw data directory
        save_dir: Path to save processed data
        scaler_path: Path to save scalers
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting preprocessing with {num_workers} workers")
    logger.info(f"Input directory: {root_dir}")
    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Scaler directory: {scaler_path}")

    # Signal configuration
    signal_max_satellites = {
        'GPS_L1_CA': 16,  'BDS_B1_I': 12,
        # 'GAL_E1_C_P': 12, 'GAL_E5A_Q': 11, 'GPS_L5_Q': 8, 'BDS_B2A_P': 8,'GLO_G1_CA': 14,
    }

    signal_order = {
        'GPS_L1_CA': 1,  'BDS_B1_I': 2,
        # 'GAL_E1_C_P': 4, 'GAL_E5A_Q': 5, 'GPS_L5_Q': 6, 'BDS_B2A_P': 7,'GLO_G1_CA': 2,
    }

    # Traces to exclude
    exclude_traces = [
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
    exclude_traces = [os.path.normpath(trace) for trace in exclude_traces]

    # Step 1: Calculate global statistics
    logger.info("Collecting global statistics for normalization...")
    scalers = collect_global_statistics(root_dir, exclude_traces)

    # Save scalers
    save_scalers(scalers, scaler_path)

    # Step 2: Find all subdirectories to process
    logger.info("Finding all subdirectories to process...")
    subdirs = find_all_subdirs(root_dir)
    logger.info(f"Found {len(subdirs)} subdirectories to process")

    # Step 3: Process traces in parallel
    logger.info("Processing traces in parallel...")
    all_trace_data = {}

    # Create a partial function with fixed parameters
    process_func = partial(
        process_trace,
        exclude_traces=exclude_traces,
        signal_max_satellites=signal_max_satellites,
        signal_order=signal_order,
        scalers=scalers
    )

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_subdir = {executor.submit(process_func, subdir): subdir for subdir in subdirs}

        # Process results as they complete
        for future in tqdm.tqdm(as_completed(future_to_subdir), total=len(subdirs), desc="Processing traces"):
            subdir = future_to_subdir[future]
            try:
                result = future.result()
                if result:
                    for trace_device_id, data in result.items():
                        if trace_device_id not in all_trace_data:
                            all_trace_data[trace_device_id] = {
                                'features': data['features'],
                                'labels': data['labels'],
                                'true_corrections': data['true_corrections'],
                                'wls_ecef': data['wls_ecef']
                            }
                        else:
                            all_trace_data[trace_device_id]['features'].extend(data['features'])
                            all_trace_data[trace_device_id]['labels'].extend(data['labels'])
                            all_trace_data[trace_device_id]['true_corrections'].extend(data['true_corrections'])
                            all_trace_data[trace_device_id]['wls_ecef'].extend(data['wls_ecef'])
            except Exception as e:
                logger.error(f"Error processing {subdir}: {e}")

    # Step 4: Save processed data
    logger.info(f"Saving processed data for {len(all_trace_data)} traces...")
    save_trace_data(all_trace_data, save_dir)

    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    # Example usage:
    path_to_raw_data = r'G:\毕业论文\data\processed_data'
    path_to_save_processed_data = r'G:\毕业论文\data\new_stand_np'
    scaler_save_path = r'G:\毕业论文\gnss\by_scalers'

    # Use 75% of available CPUs by default
    num_workers = max(10, int(multiprocessing.cpu_count() * 0.75))
    print(num_workers)

    preprocess_data(
        path_to_raw_data,
        path_to_save_processed_data,
        scaler_save_path,
        num_workers=num_workers
    )