import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 新数据集名称
datasets = ['flickr30k', 'coco', 'pet', 'lexica', 'simpsons', 'wikiart', 'kream', 'sketch']

# Fine-tune 结果（红色虚线）
fine_tune = [74.20, 43.68, 16.41, 62.32, 28.50, 53.77, 51.65, 9.95]

# w/o fine-tune 结果（Task 0）
w_o_fine_tune = [35.80, 10.40, 2.02, 3.99, 2.61, 4.62, 9.73, 1.10]

# our method de 各个任务的准确率
methods = {
    'our method': [[80.90, 82.00, 83.60, 83.20, 83.10, 85.00, 85.30, 84.40], #flick
                   [36.86, 43.04, 50.60, 51.44, 54.86, 57.80, 58.72, 56.92], #coco
                   [9.46, 11.37, 19.65, 19.11, 19.43, 20.33, 19.81, 19.73]], # pet
    'EWC': [[78.00, 79.50, 80.00, 80.50, 81.00, 82.00, 82.50, 81.50],
            [35.00, 41.00, 48.00, 49.00, 52.00, 55.00, 56.00, 54.00],
            [8.00, 10.00, 18.00, 17.00, 18.00, 19.00, 18.00, 18.00]],
    'POD': [[75.00, 76.50, 77.00, 77.50, 78.00, 79.00, 79.50, 78.50],
            [30.00, 36.00, 45.00, 46.00, 48.00, 50.00, 51.00, 49.00],
            [7.00, 9.00, 16.00, 15.00, 16.00, 17.00, 16.00, 16.00]],
    'ZSCL': [[76.00, 77.50, 78.00, 78.50, 79.00, 80.00, 80.50, 79.50],
             [32.00, 38.00, 46.00, 47.00, 50.00, 52.00, 53.00, 51.00],
             [6.50, 8.00, 14.00, 13.00, 14.00, 15.00, 14.50, 14.00]],
    'MOE-CL': [[79.00, 80.00, 81.00, 81.50, 82.00, 83.00, 83.50, 82.50],
               [34.00, 40.00, 47.00, 48.00, 51.00, 53.00, 54.00, 52.00],
               [8.50, 10.50, 17.50, 16.50, 17.00, 18.00, 17.50, 17.00]],
    'Mod-X': [[81.00, 82.50, 83.50, 84.00, 84.50, 85.00, 85.50, 84.50],
              [37.00, 44.00, 51.00, 52.00, 55.00, 58.00, 59.00, 57.00],
              [9.00, 11.00, 20.00, 19.00, 20.00, 21.00, 20.00, 20.00]]
}

# 任务编号（X轴）
tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 任务对应的数据集名称
task_names = ['w/o fine-tune', 'flickr30k', 'coco', 'pet', 'lexica', 'simpsons', 'wikiart', 'kream', 'sketch']

# 创建1行8列的子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    # 获取当前数据集的准确率
    accuracies = [
        w_o_fine_tune[i],
        *methods['our method'][i]
    ]

    # 绘制蓝色曲线（我们的方法）
    axs[i].plot(tasks, accuracies, color='#d62728', alpha=0.8, label='Our Method', marker='*', markersize=8)

    # 绘制红色虚线（Fine-tune 结果）
    axs[i].axhline(y=fine_tune[i], color='#ff7f0e', linestyle='--', label='Fine-tune', linewidth=1)

    # 设置标题和标签
    axs[i].set_title(datasets[i], fontsize=16)
    axs[i].set_ylabel('image-text R@1', fontsize=16)
    axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置X轴刻度和标签，并旋转60度
    axs[i].set_xticks(tasks)
    axs[i].set_xticklabels(task_names, rotation=60, fontsize=16)

    axs[i].tick_params(axis='y', labelsize=16)
    axs[i].axvline(x=i + 1, color='gray', linestyle='--', linewidth=1)

# 调整布局
plt.tight_layout()

# 获取图例句柄和标签
handles, labels = axs[0].get_legend_handles_labels()

# 添加图例到图形上方
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=14, ncol=2)

# 保存并显示图形
plt.savefig('fig2.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()
