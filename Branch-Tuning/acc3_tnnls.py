import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数据
methods = ['SimCLR', 'BYOL', 'BarlowTwins', 'MoCoV2+']
# steps = ['step0', 'step1', 'step2', 'step3', 'step4']
steps = ['1', '2', '3', '4', '5']
joint_train_values = [65.8, 70.5, 70.9, 69.9]
finetuning = [
    [44.8, 45.7, 47.22, 47.91, 48.92],
    [52, 51.1, 47.3, 48.5, 52.7],
    [51.8, 53.1, 53.7, 54, 54.3],
    [50.17, 49.32, 47.96, 47.2, 47.3]
]
branch_tuning = [
    [44.8, 47.79, 49.24, 51.6, 53.21],
    [52, 54.04, 56.35, 56.84, 57.92],
    [51.8, 54.36, 54.59, 56.25, 57.15],
    [50.17, 50.83, 51.53, 51.7, 52.68]
]
finetuning_kd = [
    [44.8, 53, 55.1, 57, 58.3],
    [52, 58.9, 60.9, 61.1, 62.2],
    [51.8, 58, 59.6, 59.65, 60.4],
    [50.17, 55.3, 55.9, 58.2, 59.5]
]
branch_tuning_kd = [
    [44.8, 54.2, 55.6, 58, 59],
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
colors = ['blue', 'orange', 'green', 'purple']
colors = ['lightskyblue', 'orange', 'green', 'coral']

for i, method in enumerate(methods):
    axs[i].plot(steps, finetuning[i], label='Fine-Tuning', color=colors[0], linestyle='-', marker='o', markersize=6, )
    axs[i].plot(steps, branch_tuning[i], label='Branch-Tuning', color=colors[1], linestyle='-', marker='o',
                markersize=6, )
    axs[i].plot(steps, finetuning_kd[i], label='CaSSLe', color=colors[2], linestyle='-', marker='o',
                markersize=6, )
    axs[i].plot(steps, branch_tuning_kd[i], label='CaSSLe +BT', color=colors[3], linestyle='-', marker='o',
                markersize=6, )
    # axs[i].plot(steps, finetuning[i], label='Fine-tuning', color=colors[0], linestyle='-')
    # axs[i].plot(steps, branch_tuning[i], label='Branch-tuning', color=colors[1], linestyle='-')
    # axs[i].plot(steps, finetuning_kd[i], label='Fine-tuning+KD', color=colors[2], linestyle='-')
    # axs[i].plot(steps, branch_tuning_kd[i], label='Branch-tuning+KD', color=colors[3], linestyle='-')
    # axs[i].set_title(f"{method} on CIFAR-100")
    axs[i].set_title(f"{method}", fontsize=fontsize2)
    axs[i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    # axs[i].set_ylabel('Accuracy (%)')
    axs[i].set_xlabel('Stage', fontsize=fontsize2)

    # 添加Joint train的红色棱形标记
    # axs[i].plot(steps[-1], joint_train_values[i], marker='D', markersize=8, color='red',
    #             label='Joint train' if i == 0 else None)
    axs[i].set_xticklabels(steps, fontsize=fontsize1)
    axs[0].tick_params(axis='y', labelsize=fontsize1)
# 显示图像
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize0)
# plt.tight_layout()
plt.show()
# fig.savefig('Branch-tuning-acc.pdf')
# fig.savefig('Branch-tuning-acc.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
fig.savefig('Branch-tuning-acc.pdf', bbox_inches='tight')
