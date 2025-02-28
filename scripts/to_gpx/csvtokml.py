import pandas as pd
import simplekml
import os
import threading


def csv_to_kml(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a KML object
    kml = simplekml.Kml()

    # Define styles for alternating colors
    style_red = simplekml.Style()
    style_red.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'

    style_blue = simplekml.Style()
    style_blue.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-circle.png'

    # Iterate over each row in the CSV
    for index, row in df.iterrows():
        # Create a point
        pnt = kml.newpoint()
        pnt.coords = [(row['LongitudeDegrees'], row['LatitudeDegrees'])]
        pnt.description = f"Timestamp: {row['UnixTimeMillis']}"

        # Alternate colors every 10 points
        if index % 10 == 0:
            pnt.name = f" {index // 10 + 1}"
        if index // 10 % 2 == 0:
            pnt.style = style_red
        else:
            pnt.style = style_blue
    parts_of_path = csv_file.split(os.sep)
    date_folder = parts_of_path[-3]  # 根据文件路径修改索引以指向正确的文件夹名称
    kml_filename = os.path.join(os.path.dirname(csv_file), f"{date_folder}.kml")

    # Save the KML file
    kml.save(kml_filename)


# 使用函数
''' test use
csv_path = "/Users/park/PycharmProjects/gnss/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl/ground_truth.csv"
csv_to_kml(csv_path,
           '/Users/park/PycharmProjects/gnss/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl/output.kml')

os.path='/Users/park/PycharmProjects/gnss/sdc2023/train'
'''


def find_csv_files(root_folder, filename="ground_truth.csv"):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            yield os.path.join(root, filename)


def main():
    root_folder = "sdc2023/train"  # 根目录，根据实际情况进行修改
    threads = []

    for csv_file in find_csv_files(root_folder):
        thread = threading.Thread(target=csv_to_kml, args=(csv_file,))
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
