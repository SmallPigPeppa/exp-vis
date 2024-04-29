import matplotlib.pyplot as plt
import json
import os
import re

if __name__ == '__main__':
    # 加载配置文件
    config_file = 'fig2-new-new.json'
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


    def plot_dataset(ax, dataset_i, num_categories):
        for method, settings in methods.items():
            data = config["datasets"][dataset_i][method]
            if dataset_i == '5-tasks':
                num_categories = num_categories
                ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])
            elif dataset_i == '10-tasks':
                num_categories = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
                ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])

        ax.set_xlabel('number of Classes', fontsize=fontsize0)
        ax.set_ylabel('Task Acc', fontsize=fontsize0)
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)
        minor_ticks = [30, 50, 70, 90]  # 这些是10步中的额外刻度
        ax.set_xticks(minor_ticks, minor=True)
        # ax.legend(loc='upper right', fontsize=fontsize2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=fontsize2)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figuresize)
    fig.subplots_adjust(wspace=spacing)

    plot_dataset(axs, "5-tasks", [20, 40, 60, 80, 100])

    plt.tight_layout()
    fig.savefig(f'fig2.pdf')
    plt.show()
