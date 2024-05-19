import matplotlib.pyplot as plt
import numpy as np


def plot_bar_chart():
    # 数据
    epochs = ['1 epoch', '2 epochs', '3 epochs', '5 epochs', '10 epochs', '20 epochs']
    resolutions = ['56', '112', '168']
    data = [
        [67.61, 82.918, 84.734],
        [77.49, 83.74, 84.41],
        [77.958, 83.744, 84.7],
        [77.958, 83.744, 84.7],
        [77.958, 83.744, 84.7],
        [77.958, 83.744, 84.7]
    ]

    # 设置柱状图参数
    n_epochs = len(epochs)
    n_resolutions = len(resolutions)
    bar_width = 0.2
    index = np.arange(n_epochs)

    # 绘制柱状图
    fig, ax = plt.subplots()

    for i in range(n_resolutions):
        plt.bar(index + i * bar_width, [data[j][i] for j in range(n_epochs)], bar_width, label=resolutions[i] + 'px')

    # 添加标签和标题
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy at Different Epochs and Resolutions')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(epochs)
    ax.legend()

    # 显示图形
    plt.show()


# 调用函数绘制图形
plot_bar_chart()
