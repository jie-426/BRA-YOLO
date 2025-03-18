import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 文件位置（根据实际路径修改）
    result_dict = {
        'YOLOv10': r'E:\desktop\yolov10-main\runs\detect\train15v10\results.csv',
        'YOLOv10-Bra': r'E:\desktop\yolov10-main\runs\detect\train17bra\results.csv',
        'YOLOv10-Bra-NWD': r'E:\desktop\yolov10-main\runs\detect\train18\results.csv',
    }

    def plot_metric(result_dict, col_index, ylabel, save_name):
        """
        绘制并保存指标曲线图
        :param result_dict: 模型名称和文件路径的字典
        :param col_index: 指定数据列的索引
        :param ylabel: y轴标签
        :param save_name: 保存的文件名
        """
        for modelname, res_path in result_dict.items():
            ext = res_path.split('.')[-1]
            if ext == 'csv':
                data = pd.read_csv(res_path, usecols=[col_index]).values.ravel()
            else:  # 处理 txt 文件
                with open(res_path, 'r') as f:
                    data = [float(d.strip().split()[col_index]) for d in f.readlines()]
            x = range(len(data))  # 以 epoch 为 x 轴
            plt.plot(x, data, label=modelname, linewidth=1)

        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(save_name, dpi=600)
        plt.show()

    # 绘制 Precision 曲线 (假设 Precision 在第 4 列)
    plot_metric(result_dict, 7, 'Precision', "precision.png")

    # 绘制 Recall 曲线 (假设 Recall 在第 5 列)
    plot_metric(result_dict, 8, 'Recall', "recall.png")
