import matplotlib.pyplot as plt
import json
import os
import re
if __name__ == '__main__':
    # 加载配置文件
    config_file = 'ab-lambda-new2.json'
    # config_file = 'byol-10tasks-new.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 从配置中获取设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    fontsize2 = config["plot_settings"]["fontsize2"]
    spacing = config["plot_settings"]["spacing"]
    figuresize = config["plot_settings"]["figuresize"]
    lambdas = config["plot_settings"]["lambdas"]
    datasets = config["plot_settings"]["datasets"]
    # dataset_name = config["plot_settings"]["dataset_name"]


    # 函数用于绘制每个数据集的图表
    def plot_dataset(ax, t, metric,linestyle):
        for dataset_i, settings in datasets.items():
            data = config["datasets"][dataset_i][t][metric]
            ax.plot(lambdas, data, label=dataset_i, color=settings["color"], linestyle=linestyle, marker='o', markerfacecolor='none', markeredgecolor=settings["color"],markersize=7,markeredgewidth=2)


        ax.set_xlabel('Lambda', fontsize=fontsize0)
        ax.set_ylabel(f'{metric.lower()} acc(%)', fontsize=fontsize0)

        # ax.grid(True)
        # ax.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)
        # for i in range(len(lambdas)):
        #     if i % 2 == 0:  # 每隔一个刻度绘制网格线
        #         ax.axvline(lambdas[i], color='gray', linestyle='--', linewidth=0.5)
        #
        ax.grid(True, which='major', axis='y',linestyle='--')  # 只在y轴上添加标准网格线
        # ax.grid(True, which='major', axis='y', linestyle='--')  # 只在y轴上添加标准网格线

        ax.tick_params(labelsize=fontsize1)
        ax.set_xticks(lambdas)
        # ax.tick_params(axis='x', rotation=90)



    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figuresize)
    # subplots_spacing = 15  # 可以根据需要调整这个值
    fig.subplots_adjust(wspace=spacing)

    # 绘制5task和10task的图表
    plot_dataset(axs, "T=5", "Avg",'-')
    # plot_dataset(axs[1], "T=5", "Last",'-')
    plot_dataset(axs, "T=10", "Avg",'--')
    # plot_dataset(axs[1], "T=10", "Last",'--')


    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize2)
    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(datasets) + 1, fontsize=fontsize2)
    handles, labels = axs.get_legend_handles_labels()

    # 筛选出与实线相关的句柄和标签
    solid_line_handles = [h for h, l in zip(handles, labels) if h.get_linestyle() == '-']
    solid_line_labels = [l for h, l in zip(handles, labels) if h.get_linestyle() == '-']

    # 第一部分图例 - 只包括实线的数据集名称
    legend1 = fig.legend(solid_line_handles, solid_line_labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
                         ncol=len(solid_line_handles), fontsize=fontsize2)
    import matplotlib.lines as mlines
    # 第二部分图例 - 任务T=5和T=10
    line1 = mlines.Line2D([], [], color='black', linestyle='--', label='5 Tasks')
    line2 = mlines.Line2D([], [], color='black', linestyle='-', label='10 Tasks')

    legend2 = axs.legend(handles=[line1, line2], fontsize=fontsize2)
    # legend3 = axs[1].legend(handles=[line1, line2],  fontsize=fontsize2)

    legend1.get_frame().set_facecolor('gray')
    legend1.get_frame().set_alpha(0.1)

    axs.set_yticks([64,67.0,70,73.0,76])

    plt.tight_layout()
    fig.savefig(f'{config_file.split(".")[0]}-alldataset.pdf', bbox_extra_artists=(legend1,), bbox_inches='tight')
    plt.show()
