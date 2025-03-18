import os

folder_path = r'E:\desktop\yolov10-main\ship\labels_with_ids\train\ship4-1\img1'  # 替换为你的文件夹路径

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            new_parts = [parts[0]] + parts[2:]
            new_lines.append(' '.join(new_parts) + '\n')
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

