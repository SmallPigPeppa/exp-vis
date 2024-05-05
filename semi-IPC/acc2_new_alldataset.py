import matplotlib.pyplot as plt
import json
import os
import re
if __name__ == '__main__':
    # 加载配置文件
    config_file = 'byol-5tasks.json'
    # config_file = 'byol-10tasks.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 从配置中获取设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    figuresize = config["plot_settings"]["figuresize"]
    num_tasks = config["plot_settings"]["num_tasks"]
    methods = config["plot_settings"]["methods"]
    # dataset_name = config["plot_settings"]["dataset_name"]


    # 函数用于绘制每个数据集的图表
    def plot_dataset(ax, dataset_i, num_categories):
        for method, settings in methods.items():
            data = config["datasets"][dataset_i][method]
            ax.plot(num_categories, data, label=method, linestyle=settings["linestyle"])

        ax.set_xlabel('number of Classes', fontsize=fontsize0)
        ax.set_ylabel('accuracy(%)', fontsize=fontsize0)
        # ax.set_title(f'{num_tasks} - {dataset_name}', fontsize=fontsize0)

        # n = re.match(r'\d+', dataset_i)
        ax.set_title(f'{dataset_i}', fontsize=fontsize0, loc="left")
        # ax.set_title(f'T={num_tasks}', fontsize=fontsize0, loc="right")
        ax.set_title(f'{num_tasks} Tasks', fontsize=fontsize0, loc="right")
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)

        # 为所有图表设置相同的主要刻度
        if dataset_i=="CUB-200":
            ax.set_xticks([40, 80, 120, 160, 200])
        else:
            ax.set_xticks([20, 40, 60, 80, 100])

        # 如果是10步的图表，添加次要刻度
        if num_tasks == 10:
            if dataset_i=="CUB-200":
                minor_ticks = [20, 60, 100, 140, 180]
            else:
                minor_ticks = [10, 30, 50, 70, 90]  # 这些是10步中的额外刻度
            ax.set_xticks(minor_ticks, minor=True)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figuresize)

    # 绘制5task和10task的图表
    if num_tasks==5:
        plot_dataset(axs[0], "CIFAR-100", [20, 40, 60, 80, 100])
        plot_dataset(axs[1], "ImageNet-100", [20, 40, 60, 80, 100])
        plot_dataset(axs[2], "CUB-200", [40, 80, 120, 160, 200])
        axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
        axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
        axs[2].plot([200], [82.7], label='JointCNN', marker='d', markersize=10, color='red')

    elif num_tasks==10:
        plot_dataset(axs[0], "CIFAR-100", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plot_dataset(axs[1], "ImageNet-100", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plot_dataset(axs[2], "CUB-200", [20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
        axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
        axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
        axs[2].plot([200], [82.7], label='JointCNN', marker='d', markersize=10, color='red')

    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                        ncol=int((len(methods) + 2) / 2),
                        # ncol=(len(methods) + 1),
                        fontsize=fontsize0)

    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
    #                     ncol=len(methods) + 1,
    #                     fontsize=fontsize1)

    plt.tight_layout()
    fig.savefig(f'{config_file.split(".")[0]}-alldataset.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    # fig.savefig(f'{config_file.split(".")[0]}-alldataset.pdf',)
    plt.show()
