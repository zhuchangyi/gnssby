import os
import pandas as pd
from simplekml import Kml
from concurrent.futures import ThreadPoolExecutor

def sanitize_filename(filename):
    """Sanitize filenames for Windows compatibility."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in filename)

def remove_unnecessary_files(base_directory):
    """
    Delete all files except 'gnss_data.csv' and 'ground_truth.csv' in the directory structure.

    Args:
        base_directory (str): Root directory to clean up.
    """
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if not (file.endswith("gnss_data.csv") or file.endswith("ground_truth.csv")):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

def csv_to_kml(csv_path, kml_path, kml_name, data_type):
    """
    Convert CSV data to a KML file.

    Args:
        csv_path (str): Path to the input CSV file.
        kml_path (str): Directory path to save the KML file.
        kml_name (str): Name of the output KML file.
        data_type (str): Type of data, either 'gnss_data' or 'ground_truth'.
    """
    try:
        data = pd.read_csv(csv_path)
        kml = Kml()

        if data_type == "gnss_data":
            if {'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters'}.issubset(data.columns):
                for index, row in data.iterrows():
                    x, y, z = row['WlsPositionXEcefMeters'], row['WlsPositionYEcefMeters'], row['WlsPositionZEcefMeters']
                    name = f"Point {index+1}"
                    kml.newpoint(name=name, coords=[(x, y, z)])
            else:
                print(f"CSV {csv_path} missing required columns for GNSS data.")
        elif data_type == "ground_truth":
            if {'LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters'}.issubset(data.columns):
                for index, row in data.iterrows():
                    lat, lon, alt = row['LatitudeDegrees'], row['LongitudeDegrees'], row['AltitudeMeters']
                    if pd.notnull(lat) and pd.notnull(lon):
                        name = f"Point {index+1}"
                        kml.newpoint(name=name, coords=[(lon, lat, alt)])
            else:
                print(f"CSV {csv_path} missing required columns for ground truth data.")

        os.makedirs(kml_path, exist_ok=True)
        sanitized_kml_name = sanitize_filename(kml_name) + ".kml"
        kml.save(os.path.join(kml_path, sanitized_kml_name))
        print(f"KML file saved: {os.path.join(kml_path, sanitized_kml_name)}")

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def process_file(file_path, root, data_type):
    """Helper function to process a single CSV file."""
    phone_model = root.split(os.sep)[-1]
    sanitized_root = sanitize_filename(root.replace(os.sep, '-'))
    kml_name = f"{data_type}+{phone_model}+{sanitized_root}"
    csv_to_kml(file_path, root, kml_name, data_type)

# Example usage
base_directory = "D:\\data\\processed_data"

# Step 1: Remove unnecessary files
remove_unnecessary_files(base_directory)

# Step 2: Process files concurrently
with ThreadPoolExecutor() as executor:
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith("gnss_data.csv") or file.endswith("ground_truth.csv"):
                file_path = os.path.join(root, file)
                data_type = "gnss_data" if "gnss_data" in file else "ground_truth"
                executor.submit(process_file, file_path, root, data_type)
