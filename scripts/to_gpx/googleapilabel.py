from xml.dom.minidom import Document
import csv
from datetime import datetime, timezone
import os
import threading


def create_gpx_doc():
    doc = Document()

    gpx = doc.createElement("gpx")
    gpx.setAttribute("version", "1.1")
    gpx.setAttribute("xmlns", "http://www.topografix.com/GPX/1/1")
    gpx.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    gpx.setAttribute("xsi:schemaLocation",
                     "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd")
    doc.appendChild(gpx)

    trk = doc.createElement("trk")
    gpx.appendChild(trk)

    trkseg = doc.createElement("trkseg")
    trk.appendChild(trkseg)

    return doc, trkseg


def add_csv_to_gpx(csv_filename, gpx_trkseg, doc):
    with open(csv_filename, newline='', encoding='ISO-8859-1') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            trkpt = doc.createElement("trkpt")
            trkpt.setAttribute("lat", str(row["LatitudeDegrees"]))
            trkpt.setAttribute("lon", str(row["LongitudeDegrees"]))

            if 'utcTimeMillis' in row:
                unix_time = row["utcTimeMillis"]
                # 创建名称标签，包含Unix时间戳
                name = doc.createElement("name")
                name_text = doc.createTextNode(unix_time)
                name.appendChild(name_text)
                trkpt.appendChild(name)

                # 创建时间标签，包含ISO格式的时间
                time = doc.createElement("time")
                time_text = doc.createTextNode(
                    datetime.fromtimestamp(int(unix_time) / 1000, tz=timezone.utc).isoformat())
                time.appendChild(time_text)
                trkpt.appendChild(time)

            gpx_trkseg.appendChild(trkpt)


def save_gpx(doc, output_filename):
    with open(output_filename, 'w') as fp:
        fp.write(doc.toprettyxml(indent="  "))


def csv_to_gpx_thread(csv_filename):
    doc, trkseg = create_gpx_doc()
    add_csv_to_gpx(csv_filename, trkseg, doc)

    # 使用 os.path 模块来处理文件路径
    parts_of_path = csv_filename.split(os.sep)
    date_folder = os.path.basename(os.path.dirname(os.path.dirname(csv_filename)))
    phone_name = os.path.basename(os.path.dirname(csv_filename))

    gpx_filename = os.path.join(os.path.dirname(csv_filename), f"{date_folder}_{phone_name}.gpx")
    save_gpx(doc, gpx_filename)


def find_csv_files(root_folder, filename="ground_truth.csv"):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            yield os.path.join(root, filename)


def main():
    root_folder = '/Users/park/PycharmProjects/gnss/sdc2023/train'
    threads = []

    for csv_file in find_csv_files(root_folder):
        thread = threading.Thread(target=csv_to_gpx_thread, args=(csv_file,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
