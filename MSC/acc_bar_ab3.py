import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 输入数据
methods = ['ResNet50', 'DenseNet121', 'VGG16', 'MobileNetV2']
steps = [str(int(i)) for i in range(32, 225, 32)]

# 第一组数据
# # 第一组数据
# subnet1 = [
#     [61.83, 61.73, 61.48, 60.04, 60.48, 60.39, 60.48],  # ResNet50
#     [60.11, 60.54, 60.13, 60.15, 60.07, 60.76, 60.51],  # DenseNet121
#     [58.88, 58.50, 58.88, 57.94, 58.88, 58.06, 58.88],  # VGG16
#     [58.07, 57.80, 58.07, 58.09, 58.07, 57.96, 58.07]  # MobileNetV2
# ]
#
# subnet2 = [
#     [37.11, 70.28, 71.98, 72.99, 72.76, 72.28, 73.70],  # ResNet50
#     [36.02, 62.73, 67.03, 73.83, 73.64, 73.56, 73.83],  # DenseNet121
#     [28.35, 59.05, 68.04, 72.83, 72.51, 72.44, 72.83],  # VGG16
#     [25.74, 59.93, 64.33, 70.37, 70.06, 70.12, 70.37]  # MobileNetV2
# ]
#
# subnet3 = [
#     [34.52, 61.76, 67.98, 69.96, 74.07, 74.35, 75.06],  # ResNet50
#     [33.20, 59.80, 65.54, 69.91, 73.11, 73.61, 74.12],  # DenseNet121
#     [27.62, 53.38, 64.98, 68.69, 72.94, 72.96, 74.03],  # VGG16
#     [23.73, 51.93, 51.22, 53.93, 69.31, 70.18, 71.24]  # MobileNetV2
# ]
subnet1 = [
    [61.83, 61.73, 61.48, 60.04, 60.48, 60.39, 60.48],  # ResNet50
    [60.11, 60.54, 60.13, 60.15, 60.07, 60.76, 60.51],  # DenseNet121
    [58.88, 58.50, 58.88, 57.94, 58.88, 58.06, 58.88],  # VGG16
    [58.07, 57.80, 58.07, 58.09, 58.07, 57.96, 58.07]  # MobileNetV2
]

subnet2 = [
    [37.11, 70.28, 71.98, 72.99, 72.76, 72.28, 73.70],  # ResNet50
    [36.02, 62.73, 67.03, 73.83, 73.64, 73.56, 73.63],  # DenseNet121
    [28.35, 59.05, 68.04, 72.83, 72.51, 72.44, 72.83],  # VGG16
    [25.74, 59.93, 64.33, 70.37, 70.06, 70.12, 70.37]  # MobileNetV2
]

subnet3 = [
    [34.52, 61.76, 67.98, 69.96, 74.07, 74.35, 75.06],  # ResNet50
    [33.20, 59.80, 65.54, 69.91, 74.11, 73.96, 74.12],  # DenseNet121
    [27.62, 53.38, 64.98, 68.69, 72.94, 72.96, 74.03],  # VGG16
    [23.73, 51.93, 61.22, 63.93, 70.71, 70.88, 71.24]  # MobileNetV2
]

max_values = [[], [], [], []]

# 比较每种方法的每项，并将最大值添加到结果列表中
for i in range(len(subnet1)):
    for j in range(len(subnet1[i])):
        max_values[i].append(max(subnet1[i][j], subnet2[i][j], subnet3[i][j]))

# 第二组数据
models = ['ResNet50', 'DenseNet121', 'VGG16', 'MobileNetV2']
methods_bar = ['Vanilla', 'MST', 'MSUN']
metrics = ['Average Acc', 'Parameters', 'FLOPs']
# colors = ['purple', 'lightskyblue', 'yellowgreen']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'red']

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
import matplotlib.colors as mcolors

# 转换为rgba格式，然后设置alpha为0.8
colors_rgba = [mcolors.to_rgba(c, alpha=0.7) for c in colors]

marker_style_subnet1 = dict(marker='o', color=colors_rgba[0], linewidth=2.5, markersize=12,
                            markerfacecolor='none', markeredgecolor=colors_rgba[0], markeredgewidth=3,
                            fillstyle='full', markevery=[0])  # For subnet1, fill the marker at 32

marker_style_subnet2 = dict(marker='^', color=colors_rgba[1], linewidth=2.5, markersize=12,
                            markerfacecolor='none', markeredgecolor=colors_rgba[1], markeredgewidth=3,
                            fillstyle='full', markevery=[1, 2, 3])  # For subnet2, fill the marker at 64, 96, 128

marker_style_subnet3 = dict(marker='s', color=colors_rgba[2], linewidth=2.5, markersize=12,
                            markerfacecolor='none', markeredgecolor=colors_rgba[2], markeredgewidth=3,
                            fillstyle='full', markevery=[4, 5, 6])  # For subnet3, fill the marker at 160, 192, 224
# 绘制折线图
for i, method in enumerate(methods):
    for edge, spine in axs[0, i].spines.items():
        spine.set_edgecolor('gray')
    # axs[0, i].plot(steps, subnet1[i], label='SubNet1', marker='o', color=colors[0], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors[0],markeredgewidth=3)
    # axs[0, i].plot(steps, subnet2[i], label='SubNet2', marker='^', color=colors[1], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors[1],markeredgewidth=3)
    # axs[0, i].plot(steps, subnet3[i], label='SubNet3', marker='s', color=colors[2], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors[2],markeredgewidth=3)
    # axs[0, i].plot(steps, subnet1[i], label='SubNet1', marker='o', color=colors[0], linewidth=2.5, markersize=10)
    # axs[0, i].plot(steps, subnet2[i], label='SubNet2', marker='^', color=colors[1], linewidth=2.5, markersize=10)
    # axs[0, i].plot(steps, subnet3[i], label='SubNet3', marker='s', color=colors[2], linewidth=2.5, markersize=10)
    # axs[0, i].plot(steps, max_values[i], label='MSUN', marker='o', color=colors[3], linestyle='none',markersize=6)
    axs[0, i].plot(steps, subnet1[i], label='SubNet1', **marker_style_subnet1)
    axs[0, i].plot(steps, subnet2[i], label='SubNet2', **marker_style_subnet2)
    axs[0, i].plot(steps, subnet3[i], label='SubNet3', **marker_style_subnet3)
    axs[0, i].set_xlabel('Test Size', fontsize=fontsize2)
    axs[0, i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    axs[0, i].set_title(method, fontsize=fontsize2)
    # axs[0, i].yaxis.grid(True, linestyle='-', color='grey', alpha=.8)
    axs[0, i].yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.8)
    axs[0, i].xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.8)
    axs[0, i].tick_params(axis='y', labelsize=fontsize1)
    axs[0, i].tick_params(axis='x', labelsize=fontsize1)
    axs[0, i].set_ylim([20, 80])
    # axs[0, i].legend(loc='lower right', bbox_to_anchor=(0.95, 0.05),fontsize=fontsize1)

# 绘制柱状图

index = np.arange(len(metrics))

for j, model in enumerate(models):
    for i, method in enumerate(methods_bar):
        bars = axs[1, j].bar(index + i * bar_width, data[model][method], bar_width,
                             color=colors[i], alpha=.6, label=method)
        # 在柱状图上方添加数据
        for bar in bars:
            axs[1, j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           '%.1f' % bar.get_height(), ha='center', va='bottom', fontsize=fontsize4)

        axs[1, j].set_xticks(index + bar_width)
        axs[1, j].set_xticklabels(metrics, fontsize=fontsize3)
        axs[1, j].set_yticklabels([])
        # axs[1, i].set_xlabel('Metrics', fontsize=fontsize2)
        axs[1, j].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)


# 显示图表
handles, labels = axs[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize3)
plt.tight_layout()
plt.show()
fig.savefig('MSC-combined-ab.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
