import matplotlib.pyplot as plt
import json
import os
import re

if __name__ == '__main__':
    # 加载配置文件
    # config_file = 'byol-5tasks-new.json'
    config_file = 'fs.json'
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
            if method in config["datasets"][dataset_i].keys():
                data = config["datasets"][dataset_i][method]
                ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])

        ax.set_xlabel('number of Classes', fontsize=fontsize0)
        ax.set_ylabel('accuracy(%)', fontsize=fontsize0)
        ax.set_title(f'{dataset_i}', fontsize=fontsize0, loc="left")
        if dataset_i == "CUB-200":
            num_tasks = 11
        else:
            num_tasks = 9
        ax.set_title(f'{num_tasks} Tasks', fontsize=fontsize0, loc="right")
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)

        if dataset_i == "CUB-200":
            minor_ticks = [110, 130, 150, 170, 190]
        else:
            minor_ticks = [65, 75, 85, 95]  # 这些是10步中的额外刻度
        ax.set_xticks(minor_ticks, minor=True)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figuresize)
    # subplots_spacing = 15  # 可以根据需要调整这个值
    fig.subplots_adjust(wspace=spacing)

    # 绘制5task和10task的图表

    plot_dataset(axs[0], "CUB-200", [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    plot_dataset(axs[1], "CIFAR-100", [60, 65, 70, 75, 80, 85, 90, 95, 100])
    plot_dataset(axs[2], "miniImageNet", [60, 65, 70, 75, 80, 85, 90, 95, 100])
    axs[0].plot([200], [76.5], label='JointCNN', marker='d', markersize=10, color='red')
    axs[1].plot([100], [83.5], label='JointCNN', marker='d', markersize=10, color='red')
    axs[2].plot([100], [79.0], label='JointCNN', marker='d', markersize=10, color='red')

    handles, labels = axs[0].get_legend_handles_labels()
    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.3),
    #                     ncol=int((len(methods) + 2) / 2),
    #                     # ncol=(len(methods) + 1),
    #                     fontsize=fontsize0)
    legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize2)

    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_alpha(0.1)

    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
    #                     ncol=len(methods) + 1,
    #                     fontsize=fontsize1)

    plt.tight_layout()
    fig.savefig(f'{config_file.split(".")[0]}-alldataset.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    # fig.savefig(f'{config_file.split(".")[0]}-alldataset.pdf',)
    plt.show()
