import matplotlib.pyplot as plt
import json
import os
import re

if __name__ == '__main__':
    # 加载配置文件
    config_file = 'IPC-rebuttal.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 从配置中获取设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    fontsize2 = config["plot_settings"]["fontsize2"]
    spacing = config["plot_settings"]["spacing"]
    figuresize = config["plot_settings"]["figuresize"]
    num_tasks = config["plot_settings"]["num_tasks"]
    methods = config["plot_settings"]["methods"]


    # dataset_name = config["plot_settings"]["dataset_name"]

    # 函数用于绘制每个数据集的图表
    def plot_dataset(ax, dataset_i, num_categories):
        for method, settings in methods.items():
            data = config["datasets"][dataset_i][method]
            if method == '5-tasks':
                num_categories = [50, 60, 70, 80, 90, 100]
                ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])
            elif method == '10-tasks':
                num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
                ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])

        ax.set_xlabel('number of Classes', fontsize=fontsize0)
        # ax.set_ylabel('accuracy(%)', fontsize=fontsize0)
        ax.set_ylabel(dataset_i, fontsize=fontsize0)
        # ax.set_title(f'{dataset_i}', fontsize=fontsize0, loc="left")
        # ax.set_title(f'T={num_tasks}', fontsize=fontsize0, loc="right")
        # ax.set_title(f'{num_tasks} Tasks', fontsize=fontsize0, loc="right")
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)
        minor_ticks = [55, 65, 75, 85, 95]  # 这些是10步中的额外刻度
        ax.set_xticks(minor_ticks, minor=True)
        legend = ax.legend(fontsize=fontsize2)
        # legend.get_frame().set_facecolor('gray')
        # legend.get_frame().set_alpha(0.1)


    # 创建图表
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figuresize)
    fig.subplots_adjust(wspace=spacing)

    plot_dataset(axs[0], "Distance", [50, 60, 70, 80, 90, 100])
    plot_dataset(axs[1], "Accuracy", [50, 60, 70, 80, 90, 100])

    handles, labels = axs[0].get_legend_handles_labels()
    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize2)
    # legend = fig.legend(handles, labels,fontsize=fontsize2)

    # legend.get_frame().set_facecolor('gray')
    # legend.get_frame().set_alpha(0.1)

    plt.tight_layout()
    # fig.savefig(f'fig1.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    fig.savefig(f'fig1.pdf')
    plt.show()
