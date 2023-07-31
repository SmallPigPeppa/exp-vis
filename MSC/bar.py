import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

methods = ['Branch1x1', 'Branch1x3', 'Branch3x3']
Model = ['Vanilla', 'Multi-scale Train', 'Multi-scale Framework',]
Model = ['Average Acc', 'Parameters', 'FLOPs']

data = {
    'Branch1x1': [52.91, 56.66, 54.09, 52.2],
    'Branch1x3': [ 52.89, 57.13, 56.35, 50.66],
    'Branch3x3': [50.31, 57.86, 55.62, 49.72],
}
data_fixbn = {
    'Branch1x1': [ 51.41, 56.58, 53.75, 52.34 ],
    'Branch1x3': [53.21, 57.96, 57.15, 52.62 ],
    'Branch3x3': [51.42, 57.89, 56.42, 52.68],
}

# Set hatch linewidth
mpl.rcParams['hatch.linewidth'] = 5

fig, ax = plt.subplots(figsize=(7, 5))
markersize=10
fontsize0=14
fontsize1=16
fontsize2=20

bar_width = 0.2
opacity = 1
ax.set_ylim([45, 62])
n_models = len(Model)
index = np.arange(n_models) * (len(methods) * 2 + 1) * bar_width
index = np.arange(n_models) * (len(methods) * 2 + 2) * bar_width

colors = ['yellowgreen', 'purple', 'coral']
colors = ['lightskyblue', 'orange', 'green']
colors=['purple','lightskyblue','yellowgreen']
# ax.grid(True, linestyle='--', linewidth=0.5,zorder=0)
for model_idx, model in enumerate(Model):
    for i, method in enumerate(methods):
        # rects1 = ax.bar(index[model_idx] + i * 2 * bar_width, data[method][model_idx], bar_width, alpha=opacity,
        #                 edgecolor='black', color=colors[i], label=method if model_idx == 0 else '',zorder=3)
        rects1 = ax.bar(index[model_idx] + i  * bar_width, data[method][model_idx], bar_width, alpha=opacity,
                        edgecolor='black', color=colors[i], label=method if model_idx == 0 else '',zorder=3)

        # rects2_hatch = ax.bar(index[model_idx] + i * 2 * bar_width + bar_width, data_fixbn[method][model_idx], bar_width,
        #                       alpha=opacity, color=colors[i], hatch='//', edgecolor='white', linewidth=1, zorder=2)
        # rects2_bg = ax.bar(index[model_idx] + i * 2 * bar_width + bar_width, data_fixbn[method][model_idx], bar_width,
        #                    alpha=opacity, color='none', edgecolor='black', linewidth=1,
        #                    label=method + ' FixBN' if model_idx == 0 else '', zorder=3)
# ax.set_xticks(np.arange(n_models) * (len(methods) * 2 + 1) * bar_width + bar_width * len(methods))
# ax.set_xticklabels(Model)


# 设置网格样式
ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
# ax.xaxis.grid(False)

# 计算方法标签位置
label_positions = [index[i] + (len(methods)-0.5) * (bar_width) for i in range(len(Model))]

# 设置x轴刻度及标签
ax.set_xticks(label_positions)
ax.set_xticklabels(Model,fontsize=fontsize1)
# ax.set_xlabel('Models',fontsize=fontsize2)
# ax.set_ylabel('Accuracy(%)',fontsize=fontsize2)
ax.tick_params(axis='y', labelsize=fontsize0)


handles, labels = ax.get_legend_handles_labels()
# legend1 = ax.legend(handles[0:5:2], labels[0:5:2], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,fontsize=fontsize1)
# ax.add_artist(legend1)

# legend_elements = [Patch(facecolor='black', edgecolor='black', label='Non-Fix BN'),
#                    Patch(facecolor='black', edgecolor='white',hatch='//', label='Fix BN')]
# legend2 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=1,fontsize=fontsize1)

plt.show()
fig.savefig('Branch-tuning-ablation.pdf',bbox_inches='tight')
