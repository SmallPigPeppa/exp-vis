import matplotlib.pyplot as plt
import json
import os
import re

if __name__ == '__main__':
    # 加载配置文件
    config_file = 'ab-encoder.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 从配置中获取设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    fontsize2 = config["plot_settings"]["fontsize2"]
    spacing = config["plot_settings"]["spacing"]
    figuresize = config["plot_settings"]["figuresize"]
    methods = config["plot_settings"]["methods"]


    # dataset_name = config["plot_settings"]["dataset_name"]

    # 函数用于绘制每个数据集的图表
    def plot_dataset(ax, t, metric, linestyle):
        for method_i, settings in methods.items():
            data = config["methods"][method_i][t]
            if t == 'T=5':
                n = [20, 40, 60, 80, 100]
            else:
                n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # ax.plot(n, data, label=method_i, color=settings["color"], linestyle=linestyle, marker='none')
            ax.plot(n, data, label=method_i,  linestyle=linestyle, marker='none')

        ax.set_xlabel('number of Classes', fontsize=fontsize0)
        ax.set_ylabel(f'accuracy(%)', fontsize=fontsize0)

        ax.grid(True, which='major', linestyle='--')
        ax.tick_params(labelsize=fontsize1)
        ax.set_title('CIFAR-100', fontsize=fontsize0, loc="left")

        if t == 'T=5':
            ax.set_title('5 Tasks', fontsize=fontsize0, loc="right")
            ax.set_yticks([50, 60, 70, 80, 90])
        if t == 'T=10':
            ax.set_title('10 Tasks', fontsize=fontsize0, loc="right")
            minor_ticks = [10, 30, 50, 70, 90]
            ax.set_xticks(minor_ticks, minor=True)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figuresize)
    # subplots_spacing = 15  # 可以根据需要调整这个值
    fig.subplots_adjust(wspace=spacing)

    # 绘制5task和10task的图表
    plot_dataset(axs[0], "T=5", "Avg", '-')
    # plot_dataset(axs[1], "T=5", "Last", '-')
    # plot_dataset(axs[0], "T=10", "Avg", '--')
    plot_dataset(axs[1], "T=10", "Last", '-')

    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize2)
    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(datasets) + 1, fontsize=fontsize2)
    handles, labels = axs[0].get_legend_handles_labels()

    # 筛选出与实线相关的句柄和标签
    solid_line_handles = [h for h, l in zip(handles, labels) if h.get_linestyle() == '-']
    solid_line_labels = [l for h, l in zip(handles, labels) if h.get_linestyle() == '-']

    # 第一部分图例 - 只包括实线的数据集名称
    # legend = fig.legend(solid_line_handles, solid_line_labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
    #                     ncol=len(solid_line_handles), fontsize=fontsize2)

    legend = fig.legend(solid_line_handles, solid_line_labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
                        ncol=len(solid_line_handles)//2+1, fontsize=fontsize1)
    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize1)

    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_alpha(0.1)

    plt.tight_layout()
    fig.savefig(f'{config_file.split(".")[0]}.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
