import numpy as np
import matplotlib.pyplot as plt

# 输入数据
models = ['ResNet50', 'DenseNet121', 'VGG16', 'MobileNetV2']
methods = ['Vanilla', 'Multi-scale Train', 'Multi-scale Framework']
metrics = ['Average Acc', 'Parameters', 'FLOPs']
colors = ['purple', 'lightskyblue', 'yellowgreen']

data = {
    'ResNet50': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'Multi-scale Train': [44.40776825, 59.92047501, 65.27773285],
        'Multi-scale Framework': [56.40776825, 64.92047501, 70.27773285],
    },
    'DenseNet121': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'Multi-scale Train': [44.40776825, 59.92047501, 65.27773285],
        'Multi-scale Framework': [56.40776825, 64.92047501, 70.27773285],
    },
    'VGG16': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'Multi-scale Train': [44.40776825, 59.92047501, 65.27773285],
        'Multi-scale Framework': [56.40776825, 64.92047501, 70.27773285],
    },
    'MobileNetV2': {
        'Vanilla': [20.62220192, 38.84151077, 49.61676788],
        'Multi-scale Train': [44.40776825, 59.92047501, 65.27773285],
        'Multi-scale Framework': [56.40776825, 64.92047501, 70.27773285],
    },
}

# 创建柱状图
bar_width = 0.2
opacity = 0.9
fontsize1 = 16
fontsize0 = 10
fontsize2 = 26
index = np.arange(len(metrics))

fig, axs = plt.subplots(1, len(models), figsize=(4 * 7, 5), sharey=True)

for j, model in enumerate(models):
    for i, method in enumerate(methods):

        bars = axs[j].bar(index + i * bar_width, data[model][method], bar_width,
                          color=colors[i], alpha=opacity, label=method)
        # 在柱状图上方添加数据
        for bar in bars:
            axs[j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        '%.2f' % bar.get_height(), ha='center', va='bottom',fontsize=fontsize0)
        for edge, spine in axs[j].spines.items():
            spine.set_edgecolor('gray')

        axs[j].set_xticks(index + bar_width)
        axs[j].set_xticklabels(metrics,fontsize=fontsize1)
        # axs[j].set_title(model,fontsize=fontsize2)
        # axs[j].legend()

        # 去掉y轴的坐标
        axs[j].set_yticklabels([])

        ## 设置柱状图的底部边缘颜色
        axs[j].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        # axs[j].yaxis.grid(True, linestyle='-', linewidth=0.5)


    # 设置Y轴的标签
    # axs[j].set_ylabel('Value',fontsize=fontsize2)

# plt.legend()
plt.show()
fig.savefig('MSC-bar.pdf')
