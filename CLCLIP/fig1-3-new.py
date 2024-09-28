import matplotlib.pyplot as plt


def set_axis_limits(ax, x_data, y_data):
    # 计算x和y数据的最小值和最大值
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)

    # 扩展范围以增加可视性
    x_range = (x_max - x_min) * 1.2
    y_range = (y_max - y_min) * 1.2

    ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    # 返回更新后的轴限
    return ax.get_xlim(), ax.get_ylim()


def plot_results():
    # 方法列表按照指定的顺序
    methods = ['w/o fine-tune', 'w/ fine-tune', 'EWC', 'POD', 'ZSCL', 'MOE-CL', 'Mod-X', 'our']

    # Flickr数据
    flickr_image_text_r1 = [45.84, 75.10, 69.00, 59.00, 65.30, 64.00, 58.00, 77.99]
    flickr_zero_shot_acc = [67.73, 35, 54, 59, 57.1, 58.9, 55.3, 63.11]

    # COCO数据
    coco_image_text_r1 = [19.61, 43.07, 34.00, 31.50, 33.80, 24.00, 33.00, 43.12]
    coco_zero_shot_acc = [67.73, 21, 23, 33, 43.1, 45, 33.1, 65.01]

    # Pet数据
    pet_image_text_r1 = [6.15, 18.59, 10.34, 6.60, 8.88, 10.87, 9.22, 17.25]
    pet_zero_shot_acc = [67.73, 21, 34, 39, 47.1, 48.9, 45.3, 65.01]

    # 计算图形大小（以厘米为单位）
    plot_width_cm = 8  # 每个子图的宽度
    plot_height_cm = 8  # 每个子图的高度
    spacing_cm = 4  # 子图之间的间距
    total_width_cm = 3 * plot_width_cm + 2 * spacing_cm
    total_height_cm = plot_height_cm
    figsize_inches = (total_width_cm / 2.54, total_height_cm / 2.54)  # 转换为英寸
    alpha = 0.8

    # 创建图形和子图
    fig, axs = plt.subplots(1, 3, figsize=figsize_inches)

    # 计算子图之间的间距
    total_width_inch = fig.get_figwidth()
    spacing_inch = spacing_cm / 2.54
    axes_width_inch = (total_width_inch - 2 * spacing_inch) / 3
    wspace = spacing_inch / axes_width_inch
    fig.subplots_adjust(wspace=wspace)

    # 设置标记和颜色
    markers = ['s', 's', 'o', '^', 'p', 'v', 'h', '*']
    # colors = ['green', 'lime', 'cyan', 'orange', 'blue', 'purple', 'magenta', 'red']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#d62728', ]

    # 绘制Flickr结果（第一个子图）
    for i, method in enumerate(methods):
        axs[0].scatter(flickr_image_text_r1[i], flickr_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], alpha=alpha)

    flickr_xlim, flickr_ylim = set_axis_limits(axs[0], flickr_image_text_r1, flickr_zero_shot_acc)
    axs[0].plot([flickr_xlim[0], flickr_xlim[1]], [flickr_ylim[0], flickr_ylim[1]], '--', color='gray')
    # axs[0].set_title('Task1 Flickr30K', fontsize=12)
    axs[0].set_title('Task1', loc='left', fontsize=12)
    axs[0].set_title('Flickr30K', loc='right', fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].set_xlabel('Task1 Image-Text R@1', fontsize=11)
    axs[0].set_ylabel('ImageNet Zero-Shot Acc', fontsize=11)

    # 绘制COCO结果（第二个子图）
    for i, method in enumerate(methods):
        axs[1].scatter(coco_image_text_r1[i], coco_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], alpha=alpha)

    coco_xlim, coco_ylim = set_axis_limits(axs[1], coco_image_text_r1, coco_zero_shot_acc)
    axs[1].plot([coco_xlim[0], coco_xlim[1]], [coco_ylim[0], coco_ylim[1]], '--', color='gray')
    # axs[1].set_title('Task2 COCO', fontsize=12)
    axs[1].set_title('Task2', loc='left', fontsize=12)
    axs[1].set_title('COCO', loc='right', fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[1].set_xlabel('Task2 Image-Text R@1', fontsize=11)
    axs[1].set_ylabel('ImageNet Zero-Shot Acc', fontsize=11)

    # 绘制Pet结果（第三个子图）
    for i, method in enumerate(methods):
        axs[2].scatter(pet_image_text_r1[i], pet_zero_shot_acc[i],
                       label=method, s=100, marker=markers[i], color=colors[i], alpha=alpha)

    pet_xlim, pet_ylim = set_axis_limits(axs[2], pet_image_text_r1, pet_zero_shot_acc)
    axs[2].plot([pet_xlim[0], pet_xlim[1]], [pet_ylim[0], pet_ylim[1]], '--', color='gray')
    # axs[2].set_title('Task3 Pet', fontsize=12)
    axs[2].set_title('Task3', loc='left', fontsize=12)
    axs[2].set_title('Pet', loc='right', fontsize=12)
    axs[2].tick_params(axis='both', which='major', labelsize=10)
    axs[2].set_xlabel('Task3 Image-Text R@1', fontsize=11)
    axs[2].set_ylabel('ImageNet Zero-Shot Acc', fontsize=11)

    # 添加图例
    handles, labels = axs[2].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12, ncol=8)

    # 调整布局并保存图形
    # plt.tight_layout()
    plt.savefig('fig1.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')


# 运行函数
plot_results()
