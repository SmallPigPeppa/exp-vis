import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

# 数据集名称
datasets = ['flickr30k', 'coco', 'pet', 'lexica', 'simpsons']

# Fine-tune 结果（红色虚线）
fine_tune = [74.20, 43.68, 16.41, 62.32, 28.50]

# w/o fine-tune 结果（Task 0）
w_o_fine_tune = [35.80, 10.40, 2.02, 3.99, 2.61]

# 各个任务的准确率
task1 = [80.90, 36.86, 9.46, 22.91, 5.34]
task2 = [82.00, 43.04, 11.37, 27.82, 5.40]
task3 = [83.60, 50.60, 19.65, 29.30, 6.19]
task4 = [83.20, 51.44, 19.11, 52.01, 6.60]
task5 = [83.10, 54.86, 19.43, 49.17, 26.54]

# 任务编号（X轴）
tasks = [0, 1, 2, 3, 4, 5]

# 任务对应的数据集名称
task_names = ['w/o fine-tune', 'flickr30k', 'coco', 'pet', 'lexica', 'simpsons']

# 创建5个子图，1行5列
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

for i in range(len(datasets)):
    # 获取当前数据集的准确率
    accuracies = [
        w_o_fine_tune[i],
        task1[i],
        task2[i],
        task3[i],
        task4[i],
        task5[i]
    ]

    # 绘制蓝色曲线（我们的方法）
    axs[i].plot(tasks, accuracies, color='#d62728', alpha=0.8, label='Our Method', marker='*', markersize=8)

    # 绘制红色虚线（Fine-tune 结果）
    axs[i].axhline(y=fine_tune[i], color='#ff7f0e', linestyle='--', label='Fine-tune', linewidth=1)

    # 设置标题和标签
    axs[i].set_title(f"{datasets[i]} (task{i+1})",fontsize=16)

    axs[i].set_ylabel('image-text R@1',fontsize=16)
    axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置X轴刻度和标签，并旋转60度
    # axs[i].set_xticks(tasks)
    # axs[i].set_xticklabels(task_names, rotation=60, fontsize=16)

    # 设置X轴刻度为0, 1, 2, 3, 4, 5
    axs[i].set_xticks([0, 1, 2, 3, 4, 5])

    # 设置X轴刻度标签为0, 1, 2, 3, 4, 5，并旋转60度
    axs[i].set_xticklabels([0, 1, 2, 3, 4, 5], rotation=0, fontsize=16)
    axs[i].set_xlabel('tasks', fontsize=16)


    axs[i].tick_params(axis='y', labelsize=14)
    axs[i].axvline(x=i + 1, color='gray', linestyle='--', linewidth=1)
    # axs[i].grid(True, which='both',axis='y', linestyle='--', color='gray', alpha=0.5)
    # axs[i].set_facecolor(mcolors.to_rgba('yellow', alpha=0.05))

# 调整布局
plt.tight_layout()

# 获取图例句柄和标签
handles, labels = axs[0].get_legend_handles_labels()

# 添加图例到图形上方
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=14, ncol=2)

# 保存并显示图形
plt.savefig('fig2.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()
