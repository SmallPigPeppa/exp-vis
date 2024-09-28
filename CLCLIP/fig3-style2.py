import numpy as np
import matplotlib.pyplot as plt


font1=18
font2=16


def plot_zero_shot_trends(data, methods, colors, markers, new_method_data):
    tasks = np.arange(9)  # 任务从0到8
    datasets = ['ImageNet', 'CIFAR100', 'Caltech101']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for idx, dataset in enumerate(datasets):
        # 添加新的曲线
        axs[idx].plot(tasks, new_method_data[dataset], linestyle='--', color='#1f77b4', label='w/o Fine-tune',
                      linewidth=2)

        for method, color, marker in zip(methods, colors, markers):
            axs[idx].plot(tasks, data[dataset][method], marker=marker, color=color, alpha=0.6, label=method)
            # axs[idx].plot(tasks, data[dataset][method],  color=color, alpha=0.6, label=method)


        axs[idx].set_title(f'{dataset}',fontsize=font1)
        axs[idx].set_xlabel('Task Index',fontsize=font1)
        axs[idx].set_ylabel('Zero-Shot Acc',fontsize=font1)
        axs[idx].set_xticks(tasks)
        # axs[idx].grid(True)
        axs[idx].tick_params(axis='y', labelsize=font2)
        axs[idx].tick_params(axis='x', labelsize=font2)

    # 获取图例句柄和标签
    handles, labels = axs[0].get_legend_handles_labels()

    # 添加图例到图形上方
    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font2)
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=font2, ncol=8)
    # legend.get_frame().set_facecolor('gray')
    # legend.get_frame().set_alpha(0.1)

    plt.tight_layout()
    plt.savefig('fig3.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()


# 数据集和方法
methods = ['Fine-tune', 'EWC', 'POD', 'ZSCL', 'MOE-CL', 'Mod-X', 'Our Method']
colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728']
markers = ['s', 'o', '^', 'p', 'v', 'h', '*']

# 实验结果
data = {
    'ImageNet': {
        'Our Method': [67.73, 63.11, 65.31, 63.26, 63.31, 61.95, 60.53, 60.01, 59.81],
        'Fine-tune': [67.73, 45.12, 42.1, 32.03, 30.24, 28.39, 26.75, 24.5, 22.81],
        'ZSCL': [67.73, 60.22, 58.67, 56.88, 54.13, 52.42, 50.99, 48.35, 46.87],
        'Mod-X': [67.73, 58.77, 56.23, 54.54, 52.82, 50.13, 48.78, 46.01, 44.22],
        'MOE-CL': [67.73, 55.92, 53.29, 51.08, 49.75, 47.33, 45.61, 43.88, 41.05],
        'EWC': [67.73, 50.81, 48.75, 46.67, 44.22, 42.88, 40.11, 38.65, 36.5],
        'POD': [67.73, 52.12, 43.58, 41.76, 39.41, 37.09, 35.12, 33.28, 31.47],
    },
    'CIFAR100': {
        'Our Method': [67.73, 63.11, 65.31, 63.26, 63.31, 61.95, 60.53, 60.01, 59.81],
        'Fine-tune': [67.73, 45.12, 42.1, 32.03, 30.24, 28.39, 26.75, 24.5, 22.81],
        'ZSCL': [67.73, 60.22, 58.67, 56.88, 54.13, 52.42, 50.99, 48.35, 46.87],
        'Mod-X': [67.73, 58.77, 56.23, 54.54, 52.82, 50.13, 48.78, 46.01, 44.22],
        'MOE-CL': [67.73, 55.92, 53.29, 51.08, 49.75, 47.33, 45.61, 43.88, 41.05],
        'EWC': [67.73, 50.81, 48.75, 46.67, 44.22, 42.88, 40.11, 38.65, 36.5],
        'POD': [67.73, 52.12, 43.58, 41.76, 39.41, 37.09, 35.12, 33.28, 31.47],
    },
    'Caltech101': {
        'Our Method': [67.73, 63.11, 65.31, 63.26, 63.31, 61.95, 60.53, 60.01, 59.81],
        'Fine-tune': [67.73, 45.12, 42.1, 32.03, 30.24, 28.39, 26.75, 24.5, 22.81],
        'ZSCL': [67.73, 60.22, 58.67, 56.88, 54.13, 52.42, 50.99, 48.35, 46.87],
        'Mod-X': [67.73, 58.77, 56.23, 54.54, 52.82, 50.13, 48.78, 46.01, 44.22],
        'MOE-CL': [67.73, 55.92, 53.29, 51.08, 49.75, 47.33, 45.61, 43.88, 41.05],
        'EWC': [67.73, 50.81, 48.75, 46.67, 44.22, 42.88, 40.11, 38.65, 36.5],
        'POD': [67.73, 52.12, 43.58, 41.76, 39.41, 37.09, 35.12, 33.28, 31.47],
    },
}

# 新方法的数据：每个数据集在任务0的值
new_method_data = {
    'ImageNet': [67.73] * 9,
    'CIFAR100': [67.73] * 9,
    'Caltech101': [67.73] * 9,
}

# 调用绘图函数
plot_zero_shot_trends(data, methods, colors, markers, new_method_data)
