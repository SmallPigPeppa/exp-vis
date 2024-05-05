import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    # 加载配置文件
    with open('experiments/cifar100.json', 'r') as file:
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
            ax.plot(num_categories, data, label=method, color=settings["color"], linestyle=settings["linestyle"])

        ax.set_xlabel('Number of Classes', fontsize=fontsize0)
        ax.set_ylabel('Top-1 Accuracy', fontsize=fontsize0)
        ax.set_title(f'{num_tasks} - {dataset_name}', fontsize=fontsize0)
        ax.grid(True)
        ax.tick_params(labelsize=fontsize1)


    # 创建图表
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figuresize)

    # 绘制5task和10task的图表
    plot_dataset(axs[0], "5task", [20, 40, 60, 80, 100])
    plot_dataset(axs[1], "10task", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    plt.tight_layout()
    fig.savefig('cifar100-all-new.pdf')
    plt.show()
