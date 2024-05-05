import matplotlib.pyplot as plt
import json
import os

if __name__ == '__main__':
    # 加载配置文件
    config_file = 'new_old_fetril2.json'
    with open(os.path.join('experiments', config_file), 'r') as file:
        config = json.load(file)

    # 设置
    fontsize0 = config["plot_settings"]["fontsize0"]
    fontsize1 = config["plot_settings"]["fontsize1"]
    figuresize = config["plot_settings"]["figuresize"]
    methods_data = config["methods_data"]

    # 创建图表
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figuresize)
    subplot_spacing_horizontal = 40
    subplot_spacing_vertical = 0
    fig.subplots_adjust(wspace=subplot_spacing_horizontal, hspace=subplot_spacing_vertical)

    for i, (method, data) in enumerate(methods_data.items()):
        ax = axs[i // 2, i % 2]
        x_data = list(range(1, len(data["Base"]) + 1))
        ax.plot(x_data, data["Base"], label='Base', linestyle='-', marker='o', color='blue', markersize=8,
                markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='blue')
        ax.plot(x_data, data["Novel"], label='Novel', linestyle='-', marker='o', color='orange', markersize=8,
                markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='orange')

        ax.set_xlabel('Tasks', fontsize=fontsize0)
        ax.set_ylabel('Top-1 Accuracy', fontsize=fontsize0)
        ax.set_title(method, fontsize=fontsize0)
        # ax.set_title(method, fontsize=fontsize0, loc='right')
        # ax.set_title("CIFAR-100", fontsize=fontsize0, loc='left')
        ax.tick_params(labelsize=fontsize1)
        ax.grid(True)

        ax.set_yticks([0, 20, 40, 60, 80, 100])
        # minor_ticks = [0, 100]
        # ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks([1, 3, 5, 7, 9])
        minor_ticks = [2, 4, 6, 8]
        ax.set_xticks(minor_ticks, minor=True)

    # 设置图例
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize1)
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=fontsize0)
    legend.get_frame().set_facecolor('gray')
    legend.get_frame().set_alpha(0.1)

    # 保存并显示图表
    plt.tight_layout()

    fig.savefig(f's-new-old.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()
