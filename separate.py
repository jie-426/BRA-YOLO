import os
import shutil

def separate_files(input_folder, jpg_folder, txt_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    # 遍历输入文件夹中的文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查是否为文件（排除子文件夹）
        if os.path.isfile(file_path):
            # 根据扩展名分类
            if filename.lower().endswith('.jpg'):
                shutil.move(file_path, os.path.join(jpg_folder, filename))
            elif filename.lower().endswith('.txt'):
                shutil.move(file_path, os.path.join(txt_folder, filename))

# 示例用法
input_folder = "E:\desktop\yolov10-main\RSOD2/train"
jpg_folder = "E:\desktop\yolov10-main\RSOD2/jpg"
txt_folder = "E:\desktop\yolov10-main\RSOD2/txt"

separate_files(input_folder, jpg_folder, txt_folder)
