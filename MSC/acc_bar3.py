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
# metrics = ['Parameters', 'FLOPs', 'Average Acc', ]
metrics = ['Average Acc', 'FLOPs', 'Parameters']
colors = ['purple', 'lightskyblue', 'yellowgreen']

raw_data = {
    'ResNet50': {
        'Vanilla': {'Params': 25.6e6, 'FLOPs': 3.14e12, 'Average Acc': 61.07},
        'MST': {'Params': 25.6e6, 'FLOPs': 3.14e12, 'Average Acc': 66.76},
        'MSUN': {'Params': 26.0e6, 'FLOPs': 2.24e12, 'Average Acc': 71.67},
    },
    'DenseNet121': {
        'Vanilla': {'Params': 8.0e6, 'FLOPs': 2.19e12, 'Average Acc': 61.66},
        'MST': {'Params': 8.0e6, 'FLOPs': 2.19e12, 'Average Acc': 65.67},
        'MSUN': {'Params': 8.7e6, 'FLOPs': 1.62e12, 'Average Acc': 70.05},
    },
    'VGG16': {
        'Vanilla': {'Params': 138.0e6, 'FLOPs': 1.19e13, 'Average Acc': 58.40},
        'MST': {'Params': 138.0e6, 'FLOPs': 1.19e13, 'Average Acc': 66.65},
        'MSUN': {'Params': 141.0e6, 'FLOPs': 5.62e12, 'Average Acc': 69.37},
    },
    'MobileNetV2': {
        'Vanilla': {'Params': 3.5e6, 'FLOPs': 2.36e11, 'Average Acc': 57.13},
        'MST': {'Params': 3.5e6, 'FLOPs': 2.36e11, 'Average Acc': 63.71},
        'MSUN': {'Params': 3.9e6, 'FLOPs': 1.66e11, 'Average Acc': 67.20},
    }
}

def normalize_data(data):
    normalized_data = {}

    for model in data:
        model_data = data[model]
        max_params = max([model_data[variant]['Params'] for variant in model_data])
        max_flops = max([model_data[variant]['FLOPs'] for variant in model_data])
        max_acc = max([model_data[variant]['Average Acc'] for variant in model_data])

        normalized_data[model] = {}
        for variant in model_data:
            normalized_data[model][variant] = {}
            normalized_data[model][variant]['Average Acc'] = model_data[variant]['Average Acc'] / max_acc
            normalized_data[model][variant]['FLOPs'] = model_data[variant]['FLOPs'] / max_flops
            normalized_data[model][variant]['Params'] = model_data[variant]['Params'] / max_params

    return normalized_data


normalized_data = normalize_data(raw_data)
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
import matplotlib.colors as mcolors
colors_rgba = [mcolors.to_rgba(c, alpha=0.8) for c in colors]
# 绘制折线图
for i, method in enumerate(methods):
    for edge, spine in axs[0, i].spines.items():
        spine.set_edgecolor('gray')
    # axs[0, i].plot(steps, vanilla[i], label='Vanilla', marker='o', color=colors_rgba[0], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors_rgba[0],markeredgewidth=3)
    # axs[0, i].plot(steps, mst[i], label='MST', marker='^', color=colors_rgba[1], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors_rgba[1],markeredgewidth=3)
    # axs[0, i].plot(steps, msun[i], label='MSUN', marker='s', color=colors_rgba[2], linewidth=2.5, markersize=15,markerfacecolor='none', markeredgecolor=colors_rgba[2],markeredgewidth=3)
    axs[0, i].plot(steps, vanilla[i], label='Vanilla', marker='o', color=colors_rgba[0])
    axs[0, i].plot(steps, mst[i], label='MST', marker='^', color=colors_rgba[1])
    axs[0, i].plot(steps, msun[i], label='MSUN', marker='s', color=colors_rgba[2])
    axs[0, i].set_xlabel('Test Size', fontsize=fontsize2)
    axs[0, i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    axs[0, i].set_title(method, fontsize=fontsize2)
    axs[0, i].yaxis.grid(True, linestyle='--',linewidth=1., which='major', color='grey', alpha=.8)
    axs[0, i].xaxis.grid(True, linestyle='--',linewidth=1., which='major', color='grey', alpha=.8)
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
        bars = axs[1, j].bar(index + i * bar_width, normalized_data[model][method].values(), bar_width,
                             color=colors[i], alpha=0.7, label=method)
        # 在柱状图上方添加数据
        for bar in bars:
            # pass
            # a = normalized_data[model][method].values()
            # axs[1, j].text(normalized_data[model][method].values() + bar.get_width() / 2, bar.get_height(),
            #                '%.1f' % bar.get_height(), ha='center', va='bottom', fontsize=fontsize4)
            axs[1, j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           '%.1f' % bar.get_height(), ha='center', va='bottom', fontsize=fontsize4)

        axs[1, j].set_xticks(index + bar_width)
        axs[1, j].set_xticklabels(metrics, fontsize=fontsize5)
        axs[1, j].set_yticklabels([])
        # axs[1, i].set_xlabel('Metrics', fontsize=fontsize2)
        axs[1, j].yaxis.grid(True, linestyle='--',linewidth=1.5, which='major', color='grey', alpha=.5)
        # axs[1, j].set_ylim([0, 1.25])
        axs[1, 0].set_ylim([0.65, 1.2])
        axs[1, 1].set_ylim([0.65, 1.2])
        axs[1, 2].set_ylim([0.45, 1.2])
        axs[1, 3].set_ylim([0.65, 1.2])

# 显示图表
handles, labels = axs[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize3)
plt.tight_layout()
plt.show()
# fig.savefig('MSC-combined.pdf')
fig.savefig('MSC-acc.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
