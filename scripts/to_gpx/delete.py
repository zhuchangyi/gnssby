import os
import glob

def delete_gpx_files(root_folder):
    # 搜索指定目录及其所有子目录下的所有 .gpx 文件
    for gpx_file in glob.glob(os.path.join(root_folder, '**/*.gpx'), recursive=True):
        try:
            os.remove(gpx_file)  # 删除文件
            print(f"Deleted {gpx_file}")  # 打印已删除的文件名
        except OSError as e:
            print(f"Error: {gpx_file} : {e.strerror}")  # 如果有错误，打印错误信息

def main():
    root_folder = '/Users/park/PycharmProjects/gnss/sdc2023/train'  # 设置要搜索的根目录
    delete_gpx_files(root_folder)

if __name__ == "__main__":
    main()
