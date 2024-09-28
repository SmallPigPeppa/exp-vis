import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 新数据集名称
datasets = ['flickr30k', 'coco', 'pet', 'lexica', 'simpsons', 'wikiart', 'kream', 'sketch']

# Fine-tune 结果（红色虚线）
fine_tune = [74.20, 43.68, 16.41, 62.32, 28.50, 53.77, 51.65, 9.95]

# w/o fine-tune 结果（Task 0）
w_o_fine_tune = [35.80, 10.40, 2.02, 3.99, 2.61, 4.62, 9.73, 1.10]

# 各个方法的准确率
# 各个方法的准确率
methods = {
    'EWC': [[69.86, 58.64, 40.53, 43.39, 43.56, 43.66, 45.24, 45.07],  # flickr30k
            [15.53, 35.42, 34.77, 30.44, 33.30, 35.45, 33.89, 33.73],  # coco
            [2.88, 7.43, 13.61, 10.84, 12.20, 10.35, 11.07, 10.88]],  # pet
    # 'POD': [[25.10, 60.51, 65.89, 68.56, 68.85, 68.74, 70.68, 70.36],  # flickr30k
    #         [7.64, 28.81, 32.70, 38.32, 40.03, 42.29, 45.18, 45.58],  # coco
    #         [1.98, 6.28, 8.97, 11.44, 11.62, 11.43, 12.92, 11.67]],  # pet
    'ZSCL': [[64.59, 64.72, 59.18, 58.20, 58.03, 58.07, 53.79, 51.89],  # flickr30k
             [22.13, 30.73, 30.58, 30.46, 30.98, 28.51, 28.39, 28.26],  # coco
             [2.27, 2.94, 9.08, 6.28, 5.46, 4.22, 4.25, 3.52]],  # pet
    'MOE-CL': [[69.90, 60.09, 57.68, 54.72, 50.33, 50.29, 47.02, 42.24],  # flickr30k
               [29.86, 34.80, 24.82, 25.88, 23.78, 23.27, 21.59, 21.61],  # coco
               [4.52, 7.36, 12.42, 9.11, 8.26, 8.40, 8.94, 6.74]],  # pet
    'Mod-X': [[67.31, 60.65, 56.99, 59.20, 59.48, 59.99, 51.83, 51.54],  # flickr30k
              [24.74, 33.61, 32.61, 31.74, 28.10, 27.55, 26.84, 24.25],  # coco
              [3.26, 7.12, 9.47, 7.10, 6.42, 6.21, 6.30, 4.62]],  # pet
    'our method': [[80.90, 82.00, 83.60, 83.20, 83.10, 85.00, 85.30, 84.40],  # flickr30k
                   [36.86, 43.04, 50.60, 51.44, 54.86, 57.80, 58.72, 56.92],  # coco
                   [9.46, 11.37, 19.65, 19.11, 19.43, 20.33, 19.81, 19.73]],  # pet
}

# 任务编号（X轴）
tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 创建1行3列的子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 颜色列表
colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728', ]
markers = ['o', '^', 'p', 'v', 'h', '*']  # 不同方法的标记

for i in range(3):
    # 绘制 Fine-tune 结果
    axs[i].axhline(y=fine_tune[i], color='#ff7f0e', linestyle='--', label='Fine-tune', linewidth=2)

    # 绘制各个方法的准确率
    for j, (method_name, accuracies) in enumerate(methods.items()):
        acc_i = [
            w_o_fine_tune[i],
            *accuracies[i]
        ]
        axs[i].plot(tasks, acc_i, color=colors[j], marker=markers[j], label=method_name, alpha=0.6)
        # axs[i].plot(tasks, acc_i, color=colors[j], label=method_name, alpha=0.6)

    # 设置标题和标签
    axs[i].set_title(datasets[i], fontsize=18)
    axs[i].set_ylabel('image-text R@1', fontsize=18)
    axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置X轴刻度和标签，并旋转60度
    axs[i].set_xticks(tasks)
    # axs[i].set_xticklabels(
    #     ['w/o ft', 'flickr30k', 'coco', 'pet', 'lexica', 'simpsons', 'wikiart', 'kream', 'sketch'], rotation=60,
    #     fontsize=16)

    axs[i].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8], fontsize=16)

    axs[i].tick_params(axis='y', labelsize=16)
    axs[i].axvline(x=i + 1, color='gray', linestyle='--', linewidth=2)
    axs[i].set_xlabel('Task Index', fontsize=18)

# 调整布局
plt.tight_layout()

# 获取图例句柄和标签
handles, labels = axs[0].get_legend_handles_labels()

# 添加图例到图形上方
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=16, ncol=8)

# 保存并显示图形
plt.savefig('fig2.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
# plt.show()
