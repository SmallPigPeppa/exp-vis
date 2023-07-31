import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数据
method = 'BYOL'
steps = ['1', '2', '3', '4', '5']
IPC = [82.00, 75.77, 71.43, 67.12, 63.91]
NME = [80.20, 72.58, 68.27, 64.79, 61.75]
semi_IPC = [87.05, 81.85, 77.68, 73.94, 71.22]
IPC_all = [91.55, 86.81, 82.23, 78.39, 75.95]

# 绘制子图
fig, ax = plt.subplots(figsize=(7, 5))
fontsize1 = 20
fontsize0 = 24
fontsize2 = 26
colors = ['lightskyblue', 'orange', 'green', 'coral']

ax.plot(steps, NME, label='NME', color=colors[0], linestyle='-', marker='o', markersize=6)
ax.plot(steps, IPC, label='IPC', color=colors[1], linestyle='-', marker='o', markersize=6)
ax.plot(steps, semi_IPC, label='semi-IPC', color=colors[2], linestyle='-', marker='o', markersize=6)
ax.plot(steps, IPC_all, label='IPC-all', color=colors[3], linestyle='-', marker='o', markersize=6)

# ax.set_title(f"{method}", fontsize=fontsize2)
# ax.set_title(f"CIFAR100", fontsize=fontsize2)
ax.set_ylabel('Accuracy (%)', fontsize=fontsize2)
ax.set_xlabel('Stage', fontsize=fontsize2)
ax.set_xticklabels(steps, fontsize=fontsize1)
ax.tick_params(axis='y', labelsize=fontsize1)

# 显示图像
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=fontsize0)
plt.show()
fig.savefig('BYOL-tuning-acc.pdf', bbox_inches='tight')
