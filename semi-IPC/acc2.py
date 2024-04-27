import matplotlib.pyplot as plt
import json
import os

if __name__ == '__main__':
    # 加载配置文件
    # config_file = 'cifar100.json'
    config_file = 'cifar100-byol.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 从配置中获取设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    figuresize = config["plot_settings"]["figuresize"]
    methods = config["plot_settings"]["methods"]
    dataset_name = config["plot_settings"]["dataset_name"]


    # 函数用于绘制每个数据集的图表
    def plot_dataset(ax, num_tasks, num_categories):
        for method, settings in methods.items():
            data = config["datasets"][num_tasks][method]
            ax.plot(num_categories, data, label=method, linestyle=settings["linestyle"])

        ax.set_xlabel('Number of Classes', fontsize=fontsize0)
        ax.set_ylabel('Top-1 Accuracy', fontsize=fontsize0)
        ax.set_title(f'{num_tasks} - {dataset_name}', fontsize=fontsize0)
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)

        # 为所有图表设置相同的主要刻度
        ax.set_xticks([20, 40, 60, 80, 100])

        # 如果是10步的图表，添加次要刻度
        if num_tasks == "10tasks":
            minor_ticks = [10, 30, 50, 70, 90]  # 这些是10步中的额外刻度
            ax.set_xticks(minor_ticks, minor=True)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figuresize)

    # 绘制5task和10task的图表
    plot_dataset(axs[0], "5tasks", [20, 40, 60, 80, 100])
    plot_dataset(axs[1], "10tasks", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axs[0].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')
    axs[1].plot([100], [82.7], label='JointCNN', marker='d', markersize=10, color='red')

    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
                        ncol=int((len(methods) + 2) / 2),
                        fontsize=fontsize1)

    # legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
    #                     ncol=len(methods) + 1,
    #                     fontsize=fontsize1)

    plt.tight_layout()
    # fig.savefig('cifar100-all-new.pdf')
    fig.savefig(f'{config_file.split(".")[0]}.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
