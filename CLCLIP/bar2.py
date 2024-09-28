import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

methods = ['Fine-tune', 'ZSCL', 'Mod-X', 'Our']
datasets = ['CIFAR100', 'ImageNet', 'COCO', 'Flick30K']

data = {
    'Fine-tune': [52.91, 56.66, 54.09, 52.2],
    'ZSCL': [52.89, 57.13, 56.35, 50.66],
    'Mod-X': [50.31, 57.86, 55.62, 49.72],
    'Our': [50.31, 57.86, 55.62, 49.72],
}

# 设置参数
mpl.rcParams['hatch.linewidth'] = 5
bar_width = 0.15  # 柱子宽度
opacity = 0.55    # 柱子透明度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
spacing = 0.2     # 柱子间隔

fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1行4列的子图

for model_idx, dataset in enumerate(datasets):
    for i, method in enumerate(methods):
        axs[model_idx].bar(i * (bar_width + spacing), data[method][model_idx], bar_width, alpha=opacity,
                           edgecolor='black', color=colors[i])

    # 设置标题和y轴范围
    axs[model_idx].set_xlabel(dataset, fontsize=16)
    axs[model_idx].set_ylim(45, 62)
    axs[model_idx].set_xticks([])  # 去掉x轴刻度
    axs[model_idx].grid(True, linestyle='--', linewidth=0.5)

# 只为第一个子图添加y轴标签
axs[0].set_ylabel('Zero-shot Acc (%)', fontsize=16)


# 添加图例
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=opacity) for i in range(len(methods))]

fig.legend(handles, methods, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=12)

# plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # 调整整体布局
plt.show()
fig.savefig('Branch-tuning-ablation.pdf', bbox_inches='tight')
