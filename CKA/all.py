import numpy as np
import matplotlib.pyplot as plt

# 假设数据
means_msun = [
    [0.7258, 0.9274, 0.7977, 0.7832, 0.7095],  # ResNet-50
    [0.8581, 0.8853, 0.9115, 0.9103, 0.7298],  # ResNeXt-50
    [0.8054, 0.8661, 0.9644, 0.9421, 0.8158],  # DenseNet-121
    [0.7571, 0.7546, 0.8778, 0.8306, 0.6888],  # MobileNetV2
    [0.72582, 0.92737, 0.77770, 0.78315, 0.70950],  # VGG-16
    [0.9151, 0.9471, 0.9676, 0.9054, 0.8908],  # GoogLeNet
    [0.8300, 0.8800, 0.8400, 0.8000, 0.7900],  # InceptionV3
    [0.8317, 0.8389, 0.8668, 0.8646, 0.8398]  # AlexNet
]

stds_msun = [
    [0.0164, 0.0032, 0.0599, 0.0653, 0.0071],  # ResNet-50
    [0.0439, 0.0186, 0.0084, 0.0073, 0.0134],  # ResNeXt-50
    [0.0155, 0.0234, 0.0061, 0.0071, 0.0147],  # DenseNet-121
    [0.0277, 0.0306, 0.0153, 0.0274, 0.0213],  # MobileNetV2
    [0.01637, 0.00315, 0.05995, 0.06528, 0.00709],  # VGG-16
    [0.0112, 0.0086, 0.0056, 0.0097, 0.0227],  # GoogLeNet
    [0.0330, 0.0150, 0.0250, 0.0100, 0.0200],  # InceptionV3
    [0.0324, 0.0364, 0.0113, 0.0116, 0.0166]  # AlexNet
]

means_b = [
    [0.5230, 0.8582, 0.8360, 0.7673, 0.3948],  # ResNet-50
    [0.6820, 0.8206, 0.8320, 0.8734, 0.3893],  # ResNeXt-50
    [0.6635, 0.9028, 0.9546, 0.8712, 0.6530],  # DenseNet-121
    [0.5113, 0.6342, 0.8744, 0.8922, 0.4067],  # MobileNetV2
    [0.6829, 0.8054, 0.7709, 0.6796, 0.6696],  # VGG-16
    [0.8803, 0.9564, 0.9129, 0.6906, 0.4469],  # GoogLeNet
    [0.6941, 0.8927, 0.7607, 0.7002, 0.6325],  # InceptionV3
    [0.7805, 0.7972, 0.7729, 0.7030, 0.5635]  # AlexNet
]

stds_b = [
    [0.0358, 0.0165, 0.0208, 0.0202, 0.0286],  # ResNet-50
    [0.0380, 0.0136, 0.0144, 0.0222, 0.0184],  # ResNeXt-50
    [0.0283, 0.0120, 0.0109, 0.0247, 0.0312],  # DenseNet-121
    [0.0144, 0.0201, 0.0373, 0.0487, 0.0186],  # MobileNetV2
    [0.0657, 0.0194, 0.0193, 0.0165, 0.0165],  # VGG-16
    [0.0107, 0.0068, 0.0115, 0.0337, 0.0366],  # GoogLeNet
    [0.0293, 0.0126, 0.0164, 0.0404, 0.0265],  # InceptionV3
    [0.0244, 0.0308, 0.0240, 0.0314, 0.0245]  # AlexNet
]

model_names = ['ResNet-50', 'ResNeXt-50', 'DenseNet-121', 'MobileNetV2', 'VGG-16', 'GoogLeNet', 'InceptionV3',
               'AlexNet']

# 创建 2*4 的子图
fig, axes = plt.subplots(2, 4, figsize=(4 * 6, 2 * 5))
opacity = 0.2
fontsize0 = 10
fontsize1 = 16
fontsize2 = 18
fontsize3 = 22
fontsize4 = 16
fontsize5 = 20
x = np.arange(1, len(means_msun[0]) + 1)

for i, ax in enumerate(axes.flatten()):
    for edge, spine in ax.spines.items():
        spine.set_edgecolor('gray')


    means_msun_i = means_msun[i]
    std_msun_i = stds_msun[i]

    means_b_i = means_b[i]
    std_b_i = stds_b[i]


    ax.plot(x, means_b_i, '-o', label='Baseline', color='purple', linewidth=1)
    ax.fill_between(x, np.array(means_b_i) - np.array(std_b_i), np.array(means_b_i) + np.array(std_b_i),
                    color='purple',
                    alpha=opacity, edgecolor=None)

    ax.plot(x, means_msun_i, '-o', label='MSUN', color='yellowgreen', linewidth=1)
    ax.fill_between(x, np.array(means_msun_i) - np.array(std_msun_i), np.array(means_msun_i) + np.array(std_msun_i),
                    color='yellowgreen',
                    alpha=opacity, edgecolor=None)



    # 添加轴标签和标题
    ax.set_xlabel('Block Index', fontsize=fontsize2)
    ax.set_ylabel('CKA', fontsize=fontsize2)
    ax.set_title(model_names[i], fontsize=fontsize3,loc='left')
    ax.tick_params(axis='y', labelsize=fontsize1)
    ax.tick_params(axis='x', labelsize=fontsize1)

    # 显示图例
    # ax.legend()
    ax.grid(linestyle='--',linewidth=1.5)

# 保存图形
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=fontsize3)
plt.tight_layout()
plt.savefig('result/multiple_models.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()
