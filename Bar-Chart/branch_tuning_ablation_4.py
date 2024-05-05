import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__=='__main__':
    # 输入数据
    methods = ['Branch1x1', 'Branch1x3', 'Branch3x3']
    Model = ['SimCLR', 'BYOL', 'BarlowTwins', 'MoCoV2+']

    data = {
        'Branch1x1': [51.91, 56.4, 54.09, 56.4,],
        'Branch1x3': [51.89, 58.91, 56.35, 58.91,],
        'Branch3x3': [50.31, 59.73, 55.62, 59.73, ],
    }
    data_fixbn = {
        'Branch1x1': [51.41, 56.11, 53.75, 56.11, ],
        'Branch1x3': [52.21, 59.74, 57.42, 59.74, ],
        'Branch3x3': [51.02, 60.79, 57.15, 60.79, ],
    }

    # # 创建柱状图
    # bar_width = 0.2
    # opacity = 0.5
    # index = np.arange(len(Model))
    #
    # fig, ax = plt.subplots(1, 1, figsize=(4 * 2 / 0.7, 4 * 2), layout='constrained')
    #
    # for i, method in enumerate(methods):
    #     ax.bar(index + i * bar_width, data[method], bar_width, alpha=opacity, label=method)
    #
    # ax.set_xlabel('Input Size',fontsize=20)
    # ax.set_ylabel('TOP-1 Accuracy',fontsize=20)
    # # ax.set_title('Accuracy of Different Methods on ImageNet at Different Resolutions')
    # ax.set_xticks(index + bar_width)
    # ax.set_xticklabels(Model)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(methods),fontsize=16)
    #
    # # 添加网格的横线
    # ax.yaxis.grid(True, linestyle='-', linewidth=0.5)
    # plt.setp(ax.get_xticklabels(), fontsize=16)
    # plt.setp(ax.get_yticklabels(), fontsize=16)
    #
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('MSC-imagenet.pdf')



    fig, ax = plt.subplots(figsize=(6.66, 5))
    markersize = 6
    fontsize1 = 18
    fontsize2 = 22
    ax.set_ylim([45, 62])
    bar_width = 0.2
    opacity = 0.6
    n_methods = len(methods)
    n_models = len(Model)

    index = np.arange(n_models) * (n_methods * 2 + 1) * bar_width
    colors = ['yellowgreen','yellow','coral']

    for model_idx, model in enumerate(Model):
        for i, method in enumerate(methods):
            rects1 = ax.bar(index[model_idx] + i * 2 * bar_width, data[method][model_idx], bar_width, alpha=opacity,edgecolor='black',
                            color=colors[i], label=method if model_idx == 0 else '')
            rects2 = ax.bar(index[model_idx] + i * 2 * bar_width + bar_width, data_fixbn[method][model_idx], bar_width,
                            alpha=opacity, color="none", hatch='/',edgecolor=colors[i],lw=3.,
                            label=method + ' FixBN' if model_idx == 0 else '')

    # Create legends
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles[:3], labels[:3], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles[3:5], ['No FixBN', 'FixBN'], loc='upper right', bbox_to_anchor=(1, 0.9), ncol=1)

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(np.arange(n_models) * (n_methods * 2 + 1) * bar_width + bar_width * n_methods)
    ax.set_xticklabels(Model)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    # fig.savefig('MSC-imagenet.pdf')