import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 输入数据
methods = ['ResNet50', 'DenseNet121', 'VGG16', 'MobileNetV2']
steps = [str(int(i)) for i in range(32, 225, 16)]

# 第一组数据
vanilla = [
    [19.64, 37.55, 49.22, 57.48, 63.22, 66.15, 68.84, 70.11, 71.09, 71.64, 71.96, 71.84, 75.18],  # ResNet50
    [21.07, 38.86, 51.42, 59.20, 64.38, 66.87, 69.26, 70.12, 71.07, 71.36, 71.76, 71.73, 74.51],  # DenseNet121
    [14.35, 30.51, 44.15, 54.11, 60.45, 63.48, 67.31, 68.78, 69.87, 70.40, 70.67, 70.80, 74.32],  # VGG16
    [16.16, 32.65, 44.57, 53.14, 59.20, 61.17, 65.23, 66.80, 67.58, 67.95, 68.24, 68.05, 71.97]  # MobileNetV2
]

mst = [
    [47.11, 60.65, 65.49, 67.56, 68.77, 68.80, 69.67, 69.89, 69.90, 70.02, 70.10, 70.15, 69.76],  # ResNet50
    [49.53, 58.83, 62.73, 65.15, 67.03, 67.18, 68.40, 68.78, 68.94, 69.09, 69.16, 69.31, 69.59],  # DenseNet121
    [49.96, 59.41, 63.96, 66.97, 69.07, 69.31, 69.30, 69.60, 69.64, 69.75, 69.88, 69.97, 69.68],  # VGG16
    [40.29, 55.27, 61.04, 64.53, 66.57, 66.15, 67.47, 67.76, 67.75, 67.88, 67.86, 67.94, 67.75]  # MobileNetV2
]

msun = [
    [61.83, 64.92, 70.28, 71.43, 71.98, 72.31, 72.99, 73.66, 74.07, 74.26, 74.35, 74.53, 75.06],  # ResNet50
    [60.11, 61.23, 64.27, 66.95, 69.63, 71.17, 73.83, 73.81, 74.11, 73.74, 73.61, 73.99, 74.12],  # DenseNet121
    [58.88, 60.40, 64.05, 67.10, 69.04, 70.55, 72.83, 72.98, 72.94, 72.83, 72.96, 73.19, 74.03],  # VGG16
    [58.07, 58.74, 61.93, 64.85, 67.33, 68.40, 70.37, 70.04, 70.31, 70.93, 70.88, 70.48, 71.24]  # MobileNetV2
]

# 第二组数据
models = ['ResNet50', 'DenseNet121', 'VGG16', 'MobileNetV2']
methods_bar = ['Vanilla', 'MST', 'MSUN']
metrics = ['Average Acc', 'Parameters', 'FLOPs']
colors = ['purple', 'lightskyblue', 'yellowgreen']

data = {
    'ResNet50': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'MST': [44.40776825, 59.92047501, 65.27773285],
        'MSUN': [56.40776825, 64.92047501, 70.27773285],
    },
    'DenseNet121': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'MST': [44.40776825, 59.92047501, 65.27773285],
        'MSUN': [56.40776825, 64.92047501, 70.27773285],
    },
    'VGG16': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'MST': [44.40776825, 59.92047501, 65.27773285],
        'MSUN': [56.40776825, 64.92047501, 70.27773285],
    },
    'MobileNetV2': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'MST': [44.40776825, 59.92047501, 65.27773285],
        'MSUN': [56.40776825, 64.92047501, 70.27773285],
    },
}

# 创建图表
fig, axs = plt.subplots(2, 4, figsize=(4 * 6, 2 * 5))
bar_width = 0.2
opacity = 0.9
fontsize0 = 10
fontsize1 = 16
fontsize2 = 26
fontsize3 = 22
fontsize4 = 16
fontsize5 = 20
# 绘制折线图
for i, method in enumerate(methods):
    for edge, spine in axs[0, i].spines.items():
        spine.set_edgecolor('gray')
    axs[0, i].plot(steps, vanilla[i], label='Vanilla', marker='o', color=colors[0])
    axs[0, i].plot(steps, mst[i], label='MST', marker='^', color=colors[1])
    axs[0, i].plot(steps, msun[i], label='MSUN', marker='s', color=colors[2])
    axs[0, i].set_xlabel('Test Size', fontsize=fontsize2)
    axs[0, i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    axs[0, i].set_title(method, fontsize=fontsize2)
    axs[0, i].yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.8)
    axs[0, i].tick_params(axis='y', labelsize=fontsize1)
    axs[0, i].tick_params(axis='x', labelsize=fontsize1)
    xticks = range(0, len(steps), 2)  # 每两个步长设置一个tick
    axs[0, i].set_xticks(xticks)  # 设置x轴的ticks
    axs[0, i].set_xticklabels([steps[j] for j in xticks], fontsize=fontsize1)
    # axs[0, i].legend(loc='lower right', bbox_to_anchor=(0.95, 0.05),fontsize=fontsize1)

# 绘制柱状图

index = np.arange(len(metrics))

for j, model in enumerate(models):
    for edge, spine in axs[1, j].spines.items():
        spine.set_edgecolor('gray')
    for i, method in enumerate(methods_bar):
        bars = axs[1, j].bar(index + i * bar_width, data[model][method], bar_width,
                             color=colors[i], alpha=opacity, label=method)
        # 在柱状图上方添加数据
        for bar in bars:
            a = bar.get_x()
            axs[1, j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           '%.1f' % bar.get_height(), ha='center', va='bottom', fontsize=fontsize4)

        axs[1, j].set_xticks(index + bar_width)
        axs[1, j].set_xticklabels(metrics, fontsize=fontsize5)
        axs[1, j].set_yticklabels([])
        # axs[1, i].set_xlabel('Metrics', fontsize=fontsize2)
        axs[1, j].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

# 显示图表
handles, labels = axs[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize3)
plt.tight_layout()
plt.show()
# fig.savefig('MSC-combined.pdf')
fig.savefig('MSC-acc.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
