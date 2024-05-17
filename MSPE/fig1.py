import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(data, scale=10, offset=2):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 解析数据
    models = ['Vanilla', 'FlexiViT', 'MSPE', 'origin']
    metrics = ['class_token_mean', 'class_token_var', 'patch_embed_mean', 'patch_embed_var']

    # 提取数据
    means = {model: {metric: data[metric][model] for metric in metrics} for model in models}
    vars = {model: {metric: data[metric][model] for metric in metrics} for model in models}

    # 定义绘图函数
    def plot_gaussian(ax, models, metric_mean, metric_var, title, model_offsets):
        x = np.linspace(-3 * scale, 3 * scale, 1000)
        for i, model in enumerate(models):
            mean = means[model][metric_mean] * scale + model_offsets[i]
            var = vars[model][metric_var]
            y = 1 / np.sqrt(2 * np.pi * var) * np.exp(-(x - mean) ** 2 / (2 * var))
            ax.plot(x, y, label=f"{model} (mean: {mean:.2f})")
            ax.fill_between(x, 0, y, alpha=0.3)
        ax.set_title(title)
        ax.legend()

    # 设定每个模型的均值偏移量
    offsets_class = [0, offset, 2 * offset]  # Vanilla, FlexiViT/MSPE, Origin
    offsets_patch = [0, offset, 2 * offset]  # Vanilla, FlexiViT/MSPE, Origin

    # 第一行图，类令牌分布
    plot_gaussian(axs[0, 0], ['Vanilla', 'FlexiViT', 'origin'], 'class_token_mean', 'class_token_var',
                  'Class Token Distribution: Vanilla, FlexiViT, Origin', offsets_class)
    plot_gaussian(axs[0, 1], ['Vanilla', 'MSPE', 'origin'], 'class_token_mean', 'class_token_var',
                  'Class Token Distribution: Vanilla, MSPE, Origin', offsets_class)

    # 第二行图，补丁嵌入分布
    plot_gaussian(axs[1, 0], ['Vanilla', 'FlexiViT', 'origin'], 'patch_embed_mean', 'patch_embed_var',
                  'Patch Embedding Distribution: Vanilla, FlexiViT, Origin', offsets_patch)
    plot_gaussian(axs[1, 1], ['Vanilla', 'MSPE', 'origin'], 'patch_embed_mean', 'patch_embed_var',
                  'Patch Embedding Distribution: Vanilla, MSPE, Origin', offsets_patch)

    plt.tight_layout()
    plt.show()


# 使用示例数据
data = {
    'class_token_mean': {'Vanilla': -0.050992, 'FlexiViT': -0.048826, 'MSPE': -0.053364, 'origin': -0.053925},
    'class_token_var': {'Vanilla': 0.877032, 'FlexiViT': 1.923124, 'MSPE': 1.801578, 'origin': 1.301275},
    'patch_embed_mean': {'Vanilla': -0.005856, 'FlexiViT': -0.004785, 'MSPE': -0.004493, 'origin': -0.004422},
    'patch_embed_var': {'Vanilla': 0.034332, 'FlexiViT': 0.115308, 'MSPE': 0.130930, 'origin': 0.090630}
}

plot_distributions(data, scale=10, offset=5)  # 可以调整scale和offset以优化视觉效果
