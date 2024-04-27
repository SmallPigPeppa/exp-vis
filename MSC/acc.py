import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator

# 设置风格
# sns.set(style="whitegrid")

methods = ['ResNet50', 'DenseNet121', 'VGG16','MobileNetV2']
steps = [str(int(i)) for i in range(32, 225, 16)]

vanilla = [
    [19.64, 37.55, 49.22, 57.48, 63.22, 66.15, 68.84, 70.11, 71.09, 71.64, 71.96, 71.84, 75.18], # ResNet50
    [21.07, 38.86, 51.42, 59.20, 64.38, 66.87, 69.26, 70.12, 71.07, 71.36, 71.76, 71.73, 74.51], # DenseNet121
    [14.35, 30.51, 44.15, 54.11, 60.45, 63.48, 67.31, 68.78, 69.87, 70.40, 70.67, 70.80, 74.32], # VGG16
    [16.16, 32.65, 44.57, 53.14, 59.20, 61.17, 65.23, 66.80, 67.58, 67.95, 68.24, 68.05, 71.97]  # MobileNetV2
]

mst = [
    [47.11, 60.65, 65.49, 67.56, 68.77, 68.80, 69.67, 69.89, 69.90, 70.02, 70.10, 70.15, 69.76], # ResNet50
    [49.53, 58.83, 62.73, 65.15, 67.03, 67.18, 68.40, 68.78, 68.94, 69.09, 69.16, 69.31, 69.59], # DenseNet121
    [49.96, 59.41, 63.96, 66.97, 69.07, 69.31, 69.30, 69.60, 69.64, 69.75, 69.88, 69.97, 69.68], # VGG16
    [40.29, 55.27, 61.04, 64.53, 66.57, 66.15, 67.47, 67.76, 67.75, 67.88, 67.86, 67.94, 67.75]  # MobileNetV2
]

msf = [
    [61.83, 64.92, 70.28, 71.43, 71.98, 72.31, 72.99, 73.66, 74.07, 74.26, 74.35, 74.53, 75.06], # ResNet50
    [60.11, 61.23, 64.27, 66.95, 69.63, 71.17, 73.83, 73.81, 74.11, 73.74, 73.61, 73.99, 74.12], # DenseNet121
    [58.88, 60.40, 64.05, 67.10, 69.04, 70.55, 72.83, 72.98, 72.94, 72.83, 72.96, 73.19, 74.03], # VGG16
    [58.07, 58.74, 61.93, 64.85, 67.33, 68.40, 70.37, 70.04, 70.31, 70.93, 70.88, 70.48, 71.24]  # MobileNetV2
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
colors=['purple','lightskyblue','yellowgreen']

for i, method in enumerate(methods):

    for edge, spine in axs[i].spines.items():
        spine.set_edgecolor('gray')

    axs[i].plot(steps, vanilla[i], label='Vanilla', color=colors[0], linestyle='-', marker='o', markersize=6, )
    axs[i].plot(steps, mst[i], label='Multi-scale Train', color=colors[1], linestyle='-', marker='o',
                markersize=6, )
    axs[i].plot(steps, msf[i], label='Multi-scale Framework', color=colors[2], linestyle='-', marker='o',
                markersize=6, )
    # axs[i].plot(steps, branch_tuning_kd[i], label='Branch-Tuning+KD', color=colors[3], linestyle='-', marker='o',
    #             markersize=6, )
    # axs[i].plot(steps, finetuning[i], label='Fine-tuning', color=colors[0], linestyle='-')
    # axs[i].plot(steps, branch_tuning[i], label='Branch-tuning', color=colors[1], linestyle='-')
    # axs[i].plot(steps, finetuning_kd[i], label='Fine-tuning+KD', color=colors[2], linestyle='-')
    # axs[i].plot(steps, branch_tuning_kd[i], label='Branch-tuning+KD', color=colors[3], linestyle='-')
    # axs[i].set_title(f"{method} on CIFAR-100")
    axs[i].set_title(f"{method}", fontsize=fontsize2)
    axs[i].set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=fontsize2)
    # axs[i].set_ylabel('Accuracy (%)')
    axs[i].set_xlabel('Test Size', fontsize=fontsize2)

    # 添加Joint train的红色棱形标记
    # axs[i].plot(steps[-1], joint_train_values[i], marker='D', markersize=8, color='red',
    #             label='Joint train' if i == 0 else None)

    # new_xtick_labels = ['' if j % 2 == 1 else label for j, label in enumerate(steps)]
    # axs[i].set_xticklabels(new_xtick_labels, fontsize=fontsize1)
    axs[0].tick_params(axis='y', labelsize=fontsize1)
    axs[i].tick_params(axis='x', labelsize=fontsize1)

    xticks = range(0, len(steps), 2)  # 每两个步长设置一个tick
    axs[i].set_xticks(xticks)  # 设置x轴的ticks
    axs[i].set_xticklabels([steps[j] for j in xticks], fontsize=fontsize1)  # 设置对应的标签
    axs[i].grid(True)  # 显示网格线

    # x_major_locator = MultipleLocator(2)  # 主刻度间隔为2
    # x_minor_locator = MultipleLocator(1)  # 次刻度间隔为1
    # axs[i].xaxis.set_major_locator(x_major_locator)
    # axs[i].xaxis.set_minor_locator(x_minor_locator)
    # y_major_locator = MultipleLocator(10)  # 主刻度间隔为2
    # y_minor_locator = MultipleLocator(10)  # 次刻度间隔为1
    # axs[i].yaxis.set_major_locator(y_major_locator)
    # axs[i].yaxis.set_minor_locator(y_minor_locator)


    # axs[i].grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    # axs[i].set_xticklabels([32,])




    # axs[i].grid()
    # from matplotlib.ticker import MultipleLocator

    # 在x轴上每2.5个单位添加一个次要刻度
    # axs[i].xaxis.set_minor_locator(MultipleLocator(5))
    # # 在y轴上每2.5个单位添加一个次要刻度
    # axs[i].yaxis.set_minor_locator(MultipleLocator(5))

    # 开启次要网格线
    # axs[i].grid( color='grey', linestyle='-', linewidth=0.25)

    # yticks = np.arange(20, 80, 5)
    # # new_ytick_labels = ['' if j % 2 == 1 else label for j, label in enumerate(np.arange(20, 80, 5))]
    # axs[i].set_yticks(yticks) # 设置y轴刻度
    axs[i].set_ylim([10, 80]) # 设置y轴范围
# 显示图像
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize0)
# plt.tight_layout()
plt.show()
# fig.savefig('Branch-tuning-acc.pdf')
# fig.savefig('Branch-tuning-acc.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
fig.savefig('MSC-acc.pdf', bbox_inches='tight')
