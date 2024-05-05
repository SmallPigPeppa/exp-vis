import matplotlib.pyplot as plt

# 创建数据
methods = ['Res50-B1-S3', 'Res50-B3-S3', 'Res50-B4-S3', 'Res50-B2-S2', 'Res50-B2-S4', 'Res50-B2-S5', 'Res50-B2-S3']
params = [25.60, 28.4, 31.6, 25.8, 26.2, 26.4, 26.00]
average_acc = [68.2, 72.44, 72.97, 65.79, 71.8, 71.91, 71.51]
# average_acc = [69.92, 72.38, 73.32, 67.81, 73.18, 73.58, 71.50]

# 创建颜色列表
colors = []
markers = []
sizes = [400] * len(methods)
x_a = [0.5, 0.3, -1.5, 0.4, -0.5, 0.4, 0.4]
y_a = [-0.2, -0.6, -0.6, 0.7, 0.7, 1.0, -0.6]
legend_dict = {'Res50-B2-S3': ['green', 'o'], 'Res50-B2-Si': ['red', 's'], 'Res50-Bi-S3': ['blue', '^']}

for method in methods:
    if 'B2' in method and 'S3' in method:
        colors.append('green')  # 如果"B2"和"S3"都在方法名中，颜色设为绿色
        markers.append('o')
    elif 'B2' in method:
        colors.append('red')  # 如果只有"B2"在方法名中，颜色设为红色
        markers.append('s')  # 对于"S3"，使用方形标记
    elif 'S3' in method:
        colors.append('blue')  # 如果只有"S3"在方法名中，颜色设为蓝色
        markers.append('^')
    else:
        colors.append('black')  # 如果都不在，颜色设为黑色
        markers.append('o')
# 创建图形
plt.figure(figsize=(6 * 2, 5 * 2))
fontsize0 = 35
fontsize1 = 25
fontsize2 = 30

# 为每个点添加标签
for i in range(len(methods)):
    plt.scatter(params[i], average_acc[i], s=sizes[i], marker=markers[i], c=colors[i])
    plt.text(params[i] + x_a[i], average_acc[i] + y_a[i], methods[i], fontsize=fontsize1,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    if methods[i] == 'Res50-B2-S3':  # 找到 'Res50-B2-S3' 数据点
        plt.axhline(average_acc[i], color='green', linestyle='--',linewidth=4)
        plt.axvline(params[i], color='green', linestyle='--',linewidth=4)

plt.tick_params(axis='y', labelsize=fontsize1)
plt.tick_params(axis='x', labelsize=fontsize1)
plt.title('Comparison of MSUN Configurations', fontsize=fontsize0,pad=40)
plt.xlabel('Params', fontsize=fontsize0)
plt.ylabel('Average Accuracy', fontsize=fontsize0)
plt.grid(True, linestyle='--', which='major', color='grey', alpha=.5, linewidth=3)

# 创建图例元素列表
legend_elements = [plt.Line2D([0], [0], marker=value[1], color=value[0], label=key,
                              markersize=10, linestyle='None') for key, value in legend_dict.items()]

# 添加图例
# plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.97, 0.05), fontsize=fontsize1,
           # borderaxespad=0., edgecolor='black',handlelength=2, handletextpad=2)
plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.97, 0.05), fontsize=fontsize1,
           borderaxespad=0., edgecolor='black')
# plt.legend(handles=legend_elements, loc='lower right')

plt.savefig('ab-dot.pdf')

plt.show()