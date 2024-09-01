import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")

# 数据
methods = ['Flickr30K']
# steps = ['step0', 'step1', 'step2', 'step3', 'step4']
steps = ['1', '2', '3', '4', '5']
joint_train_values = [65.8]
finetuning = [
    [37.8, 41.8, 42.4, 44.6, 45.6]
]
branch_tuning_kd = [
    [36.6, 45, 47.4, 49.6, 51.2]
]

# 绘制子图
fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 5), sharey=True)
fig.subplots_adjust(top=0.85)
markersize = 6
fontsize1 = 20
fontsize0 = 24
fontsize2 = 26
colors = ['blue', 'orange', 'green', 'purple']
colors = ['lightskyblue', 'orange', 'green', 'coral']

for i, method in enumerate(methods):
    axs.plot(steps, finetuning[i], label='Vanilla', color=colors[0], linestyle='-', marker='o', markersize=6, )
    axs.plot(steps, branch_tuning_kd[i], label='Vanilla +CKD', color=colors[3], linestyle='-', marker='o',
                markersize=6, )
    axs.set_title(f"{method}", fontsize=fontsize2)
    axs.set_ylabel('Text2Image R@1' if i == 0 else '', fontsize=fontsize2)
    axs.set_xlabel('Stage', fontsize=fontsize2)
    axs.set_xticklabels(steps, fontsize=fontsize1)
    axs.tick_params(axis='y', labelsize=fontsize1)
# 显示图像
handles, labels = axs.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize0)
# plt.tight_layout()
plt.show()
# fig.savefig('Branch-tuning-acc.pdf')
# fig.savefig('Branch-tuning-acc.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
fig.savefig('Branch-tuning-clip.pdf', bbox_inches='tight')
