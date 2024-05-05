import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数据
method = 'New Method'
steps = ['20', '40', '60', '80', '100']
semi_IPC = [87.05, 81.85, 77.68, 73.94, 71.22]
DER = [91.90, 80.30, 70.25, 63.70, 60.73]
PODNet = [91.05, 80.05, 68.22, 58.88, 53.52]
UCIR = [93.10, 73.47, 64.65, 56.51, 52.39]
ICARL = [90.90, 78.18, 66.90, 59.00, 54.52]
LWF = [90.95, 73.90, 59.65, 49.68, 42.51]

# 绘制子图
fig, ax = plt.subplots(figsize=(7, 5))
fontsize1 = 20
fontsize0 = 24
fontsize2 = 26
colors = ['lightskyblue', 'orange', 'green', 'coral', 'purple', 'brown']

ax.plot(steps, semi_IPC, label='semi-IPC', color=colors[0], linestyle='-', marker='o', markersize=6)
ax.plot(steps, DER, label='DER', color=colors[1], linestyle='-', marker='o', markersize=6)
ax.plot(steps, PODNet, label='PODNet', color=colors[2], linestyle='-', marker='o', markersize=6)
ax.plot(steps, UCIR, label='UCIR', color=colors[3], linestyle='-', marker='o', markersize=6)
ax.plot(steps, ICARL, label='ICARL', color=colors[4], linestyle='-', marker='o', markersize=6)
ax.plot(steps, LWF, label='LWF', color=colors[5], linestyle='-', marker='o', markersize=6)

# ax.set_title(f"{method}", fontsize=fontsize2)
# ax.set_title(f"CIFAR100", fontsize=fontsize2)
ax.set_ylabel('Accuracy (%)', fontsize=fontsize2)
ax.set_xlabel('Stage', fontsize=fontsize2)
ax.set_xticklabels(steps, fontsize=fontsize1)
ax.tick_params(axis='y', labelsize=fontsize1)

# 显示图像
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=fontsize0)
plt.show()
fig.savefig('New-Method-tuning-acc.pdf', bbox_inches='tight')
