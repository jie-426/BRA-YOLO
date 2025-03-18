import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def smooth_curve(data, factor=5):
    """
    使用滑动平均平滑数据曲线
    :param data: 原始数据列表或数组
    :param factor: 平滑因子，窗口大小
    :return: 平滑后的数据
    """
    if factor < 2:
        return data  # 不平滑
    return np.convolve(data, np.ones(factor) / factor, mode='valid')


def plot_metric(result_dict, col_index, ylabel, save_name, total_loss=False, dpi=150, font_size=20, color_dict=None,
                smooth_factor=5):
    """
    绘制并保存指标曲线图（支持平滑处理）
    :param result_dict: 模型名称和文件路径的字典
    :param col_index: 指定数据列的索引或索引列表（多列求和时使用）
    :param ylabel: y轴标签
    :param save_name: 保存的文件名
    :param total_loss: 是否计算总 Loss（多列求和）
    :param dpi: 保存图片的分辨率
    :param font_size: 字体大小
    :param color_dict: 颜色字典，格式为 {'model_name': 'color'}
    :param smooth_factor: 平滑因子，控制滑动平均窗口大小
    """
    plt.rc('font', family='Times New Roman', size=20)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.grid()

    if color_dict is None:
        color_dict = {}

    cmap = plt.cm.get_cmap('Set1', len(result_dict))
    default_colors = [cmap(i) for i in range(len(result_dict))]

    for i, (modelname, res_path) in enumerate(result_dict.items()):
        try:
            if res_path.endswith('.csv'):
                df = pd.read_csv(res_path)
                if total_loss:
                    data = df.iloc[:, col_index].sum(axis=1).round(5)  # 多列求和
                else:
                    data = df.iloc[:, col_index].values.ravel()
            else:
                with open(res_path, 'r') as f:
                    lines = f.readlines()
                    data = [float(re.split(r'\s+', line.strip())[col_index]) for line in lines]

            data = np.nan_to_num(data)  # 处理 NaN 和 inf

            # 平滑数据
            smoothed_data = smooth_curve(data, smooth_factor)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")
            continue

        x = range(len(smoothed_data))  # x 轴
        color = color_dict.get(modelname, default_colors[i])  # 选择颜色
        plt.plot(x, smoothed_data, label=modelname, linewidth=2, color=color)

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, len(smoothed_data), step=max(1, len(smoothed_data) // 10)+1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=dpi)
    plt.show()


# 训练结果文件路径
# Maridrone数据集
# result_dict = {
#     'YOLOv5': r'E:\desktop\yolov10-main\runs\detect\train56\results.csv',
#     'YOLOv8': r'E:\desktop\yolov10-main\runs\detect\train57\results.csv',
#     'YOLOv10': r'E:\desktop\yolov10-main\runs\detect\train58\results.csv',
#      'Faster-RCNN': r'E:\desktop\yolov10-main\runs\detect\train61\results.csv',
#         'RT-DETR': r'E:\desktop\yolov10-main\runs\detect\train59\results.csv',
#         'BRA-YOLO': r'E:\desktop\yolov10-main\runs\detect\train60\results.csv',
# }
#
# result_dict = {
#     'YOLOv5': r'E:\desktop\yolov10-main\runs\detect\train74\results.csv',
#     'YOLOv8': r'E:\desktop\yolov10-main\runs\detect\train73\results.csv',
#     'YOLOv10': r'E:\desktop\yolov10-main\runs\detect\train72\results.csv',
#      'Faster-RCNN': r'E:\desktop\yolov10-main\runs\detect\train71\results.csv',
#         'RT-DETR': r'E:\desktop\yolov10-main\runs\detect\train76\results.csv',
#         'BRA-YOLO': r'E:\desktop\yolov10-main\runs\detect\train77\results.csv',
# }

result_dict = {
    'YOLOv5': r'E:\desktop\yolov10-main\runs\detect\train79\results.csv',
    'YOLOv8': r'E:\desktop\yolov10-main\runs\detect\train80\results.csv',
    'YOLOv10': r'E:\desktop\yolov10-main\runs\detect\train81\results.csv',
     'Faster-RCNN': r'E:\desktop\yolov10-main\runs\detect\train82\results.csv',
        'RT-DETR': r'E:\desktop\yolov10-main\runs\detect\train83\results.csv',
        'BRA-YOLO': r'E:\desktop\yolov10-main\runs\detect\train84\results.csv',
}
color_dict = {
    'YOLOv5': 'purple',
    'YOLOv10': 'blue',
    'Faster-RCNN': 'black',
    'RT-DETR': 'orange',
    'BRA-YOLO': 'red'
}

# 绘制平滑的曲线
plot_metric(result_dict, 9, 'Precision', "precision.png", dpi=150, font_size=30, color_dict=color_dict, smooth_factor=5)
# plot_metric(result_dict, 8, 'Recall', "recall.png", dpi=150, font_size=30, color_dict=color_dict, smooth_factor=5)
# plot_metric(result_dict, 9, 'mAP@0.5', "mAP50.png", dpi=150, font_size=30, color_dict=color_dict, smooth_factor=5)
# plot_metric(result_dict, 10, 'mAP@0.5:0.95', "mAP50-95.png", dpi=150, font_size=30, color_dict=color_dict, smooth_factor=5)
# plot_metric(result_dict, [1, 2, 3], 'Total Loss', "total_loss.png", total_loss=True, dpi=150, font_size=30,
#             color_dict=color_dict, smooth_factor=5)
