import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def smooth_curve(image_path, window_length=21, polyorder=3):
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 边缘检测（可选，取决于你的图像）
    edges = cv2.Canny(img, 50, 150)

    # 获取曲线点坐标
    y_coords, x_coords = np.where(edges > 0)

    # 按 x 轴排序
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # 进行 Savitzky-Golay 滤波平滑曲线
    y_smooth = savgol_filter(y_sorted, window_length, polyorder)

    # 绘制原始曲线和平滑后的曲线
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.plot(x_sorted, y_sorted, 'r.', markersize=2, label='Original')
    plt.plot(x_sorted, y_smooth, 'b-', linewidth=2, label='Smoothed')
    plt.legend()
    plt.show()


# 示例调用
smooth_curve(r'E:\desktop\结果图gj\结果图\Figure_11.png')