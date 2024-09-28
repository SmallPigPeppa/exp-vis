import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

methods = ['Fine-tune', 'ZSCL', 'Mod-X', 'Our']
Datasets = ['CIFAR100', 'ImageNet', 'Caltech101', 'Food101']

data = {
    'Fine-tune': [52.91, 56.66, 54.09, 52.2],
    'ZSCL': [52.89, 57.13, 56.35, 50.66],
    'Mod-X': [50.31, 57.86, 55.62, 49.72],
    'Our': [50.31, 57.86, 55.62, 49.72],
}

# Set hatch linewidth
mpl.rcParams['hatch.linewidth'] = 5

fig, ax = plt.subplots(figsize=(10, 5))
markersize = 10
fontsize0 = 14
fontsize1 = 16
fontsize2 = 20

bar_width = 10
opacity = 0.55
ax.set_ylim([45, 62])
n_models = len(Datasets)
index = np.arange(n_models) * (len(methods) * 2 + 1) * bar_width
index = np.arange(n_models) * (len(methods) * 2 + 2) * bar_width
colors = ['lightskyblue', 'orange', 'green', 'red']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# ax.grid(True, linestyle='--', linewidth=0.5,zorder=0)
for model_idx, model in enumerate(Datasets):
    for i, method in enumerate(methods):
        # rects1 = ax.bar(index[model_idx] + i * 2 * bar_width, data[method][model_idx], bar_width, alpha=opacity,
        #                 edgecolor='black', color=colors[i], label=method if model_idx == 0 else '', zorder=3)
        rects1 = ax.bar(index[model_idx] + i * 2 * bar_width, data[method][model_idx], bar_width, alpha=opacity,
                        edgecolor='black', color=colors[i], label=method if model_idx == 0 else '', zorder=3)

# 设置网格样式
ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
# ax.xaxis.grid(False)

# 计算方法标签位置
label_positions = [index[i] + (len(methods) - 0.5) * (bar_width) for i in range(len(Datasets))]

# 设置x轴刻度及标签
ax.set_xticks(label_positions)
ax.set_xticklabels(Datasets, fontsize=fontsize1)
ax.set_xlabel('Datasets', fontsize=fontsize2)
ax.set_ylabel('Zero-shot Acc(%)', fontsize=fontsize2)
ax.tick_params(axis='y', labelsize=fontsize0)

handles, labels = ax.get_legend_handles_labels()
# legend1 = ax.legend(handles[0:5:2], labels[0:5:2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
#                     fontsize=fontsize1)

legend1 = ax.legend(handles[0:5], labels[0:5], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5,
                    fontsize=fontsize1)

ax.add_artist(legend1)

plt.show()
fig.savefig('Branch-tuning-ablation.pdf', bbox_inches='tight')
