import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(data, scale=50):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 解析数据
    models = ['Vanilla', 'FlexiViT', 'MSPE', 'origin']
    metrics = ['class_token_mean', 'class_token_var']

    # 提取数据
    means = {model: {metric: data[metric][model] * scale for metric in metrics} for model in models}
    vars = {model: {metric: data[metric][model] for metric in metrics} for model in models}

    # 定义绘图函数
    def plot_gaussian(ax, models, metric_mean, metric_var, title):
        x = np.linspace(-58, -45, 1000)
        for model in models:
            mean = means[model][metric_mean]
            var = vars[model][metric_var]
            y = 1 / np.sqrt(2 * np.pi * var) * np.exp(-(x - mean) ** 2 / (2 * var))
            ax.plot(x, y, label=f"{model} (mean: {mean / scale:.2f} scaled)")
            ax.fill_between(x, 0, y, alpha=0.3)
        ax.set_title(title)
        ax.legend()

    # 第一行图，类令牌分布
    plot_gaussian(axs[0], ['Vanilla', 'FlexiViT', 'origin'], 'class_token_mean', 'class_token_var',
                  'Class Token Distribution: Vanilla, FlexiViT, Origin')
    plot_gaussian(axs[1], ['Vanilla', 'MSPE', 'origin'], 'class_token_mean', 'class_token_var',
                  'Class Token Distribution: Vanilla, MSPE, Origin')

    # # 第二行图，补丁嵌入分布
    # plot_gaussian(axs[1, 0], ['Vanilla', 'FlexiViT', 'origin'], 'patch_embed_mean', 'patch_embed_var',
    #               'Patch Embedding Distribution: Vanilla, FlexiViT, Origin')
    # plot_gaussian(axs[1, 1], ['Vanilla', 'MSPE', 'origin'], 'patch_embed_mean', 'patch_embed_var',
    #               'Patch Embedding Distribution: Vanilla, MSPE, Origin')

    plt.tight_layout()
    plt.show()
    fig.savefig('fig1.pdf')


# 使用示例数据
data = {
    'class_token_mean': {'Vanilla': -0.050992, 'FlexiViT': -0.048826, 'MSPE': -0.053364, 'origin': -0.053925},
    'class_token_var': {'Vanilla': 0.877032, 'FlexiViT': 1.923124, 'MSPE': 1.801578, 'origin': 1.301275},
}

plot_distributions(data, scale=1000)  # Scale factor可以根据需要进行调整
