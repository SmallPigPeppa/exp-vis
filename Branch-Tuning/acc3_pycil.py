import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数据
methods = ['SimCLR', 'BYOL', 'BarlowTwins', 'MoCoV2+']
steps = ['1', '2', '3', '4', '5']

EWC = [
    [44.8, 48.7, 50.9, 51.2, 53.6],
    [52, 54.8, 55.3, 56.1, 56.4],
    [51.8, 54, 56, 56.4, 56.7],
    [50.17, 52.1, 52.4, 52.4, 52.5]
]

ER = [
    [44.8, 46.1, 48.5, 48.9, 50.3],
    [52, 52.5, 52.9, 53.5, 54.7],
    [51.8, 53.2, 53.6, 54.5, 54.6],
    [50.17, 49.32, 49.96, 50.2, 50.5]
]

DER = [
    [44.8, 46.7, 48.9, 49.8, 50.7],
    [52, 52.4, 52.4, 53.5, 54.8],
    [51.8, 53.5, 53.9, 55, 55.3],
    [50.17, 50.62, 50.96, 51, 51.1]
]

LWF = [
    [44.8, 49.3, 50.8, 51.3, 52.5],
    [52, 56, 57.1, 57.3, 58.6],
    [51.8, 54.9, 55.5, 56.1, 56.4],
    [50.17, 52.92, 52.9, 53.1, 53.2]
]

POD = [
    [44.8, 48.1, 49.22, 50.91, 51.3],
    [52, 55.1, 56.5, 57.2, 57.9],
    [51.8, 54.3, 54.7, 55.2, 55.9],
    [50.17, 51.12, 51.5, 51.9, 51.9]
]

Branch_tuning_KD = [
    [44.8, 54.2, 55.6, 58, 59.0],
    [52, 59.6, 61.7, 62.2, 63.1],
    [51.8, 59.2, 60, 60.2, 60.9],
    [50.17, 55.9, 57.8, 60.1, 60.7]
]

# 绘制子图
fig, axs = plt.subplots(1, 4, figsize=(4 * 7, 5), sharey=True)
fig.subplots_adjust(top=0.85)
markersize = 6
fontsize1 = 20
fontsize0 = 24
fontsize2 = 26
colors = ['blue', 'orange', 'green', 'purple', 'coral', 'red']

for i, method in enumerate(methods):
    axs[i].plot(steps, EWC[i], label='EWC', color=colors[0], linestyle='-', marker='o', markersize=6)
    axs[i].plot(steps, ER[i], label='ER', color=colors[1], linestyle='-', marker='o', markersize=6)
    axs[i].plot(steps, DER[i], label='DER', color=colors[2], linestyle='-', marker='o', markersize=6)
    axs[i].plot(steps, LWF[i], label='LWF', color=colors[3], linestyle='-', marker='o', markersize=6)
    axs[i].plot(steps, POD[i], label='POD', color=colors[4], linestyle='-', marker='o', markersize=6)
    axs[i].plot(steps, Branch_tuning_KD[i], label='Branch-tuning+KD', color=colors[5], linestyle='-', marker='o',
                markersize=6)

    axs[i].set_title(f"{method}", fontsize=fontsize2)
    axs[i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    axs[i].set_xlabel('Stage', fontsize=fontsize2)
    axs[i].set_xticklabels(steps, fontsize=fontsize1)
    axs[0].tick_params(axis='y', labelsize=fontsize1)

handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=fontsize0)
plt.show()
fig.savefig('updated_experiment_results.pdf', bbox_inches='tight')
