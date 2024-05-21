import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def custom_formatter(x, pos):
    return f'{int(x)}%'

title_fontsize = 26
tick_label_fontsize = 23
legend_fontsize = 21
line_width = 2
minor_tick_length = 6
l1 = 1.5
marker_size = 8

resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 238, 252, 448]
resolution_labels = [str(res) for res in resolutions]

area_accuracy = [53.068, 59.942, 75.808, 74.16, 78.312, 78.342, 81.188, 79.948, 81.02, 82.558, 85.1, 81.962, 82.104, 74.248]
bilinear_accuracy = [51.94, 38.838, 77.022, 67.09, 79.022, 66.692, 82.312, 72.232, 76.38, 83.522, 85.1, 79.81, 82.946, 76.584]
pi_resize_accuracy = [56.408, 71.016, 77.944, 79.544, 81.632, 82.508, 83.746, 83.814, 83.936, 84.7, 85.1, 85.13, 85.13, 85.1]
nearest_accuracy = [42.616, 45.718, 73.518, 67.992, 75.078, 71.628, 80.318, 77.176, 78.604, 82.058, 83.23, 81.244, 81.756, 76.728]
bicubic_accuracy = [51.356, 36.948, 77.096, 65.33, 78.672, 61.934, 82.386, 64.954, 73.68, 83.784, 84.158, 77.478, 83.024, 77.446]

fig, ax = plt.subplots(figsize=(6 / 0.7, 5.2))

ax.plot(resolution_labels, pi_resize_accuracy, marker=None, markersize=marker_size, label='PI-Resize', linewidth=l1, color='r')
# ax.scatter(resolution_labels, pi_resize_accuracy, marker='o', s=marker_size**2, label='PI-Resize', color='r')

# ax.plot(resolution_labels, area_accuracy, marker='s', markersize=marker_size, label='Area', linestyle='--', linewidth=l1, color='#1f77b4')
# ax.plot(resolution_labels, bilinear_accuracy, marker='o', markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1, color='#ff7f0e')
# ax.plot(resolution_labels, bicubic_accuracy, marker='>', markersize=marker_size, label='Bicubic', linestyle='--', linewidth=l1, color='#2ca02c')
# ax.plot(resolution_labels, nearest_accuracy, marker='^', markersize=marker_size, label='Nearest', linestyle='--', linewidth=l1, color='#9467bd')
# ax.plot(resolution_labels, area_accuracy, marker='s', markersize=marker_size, label='Area', linestyle='--', linewidth=l1,)
# ax.plot(resolution_labels, bilinear_accuracy, marker='o', markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1,)
# ax.plot(resolution_labels, bicubic_accuracy, marker='*', markersize=marker_size, label='Bicubic', linestyle='--', linewidth=l1, )
# ax.plot(resolution_labels, nearest_accuracy, marker='^', markersize=marker_size, label='Nearest', linestyle='--', linewidth=l1, )

# ax.plot(resolution_labels, area_accuracy, marker='s', markersize=marker_size, label='Area', linestyle='--', linewidth=l1, color='#1f77b4')
# ax.plot(resolution_labels, bilinear_accuracy, marker='o', markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1, color='#ff7f0e')
# ax.plot(resolution_labels, bicubic_accuracy, marker='*', markersize=marker_size, label='Bicubic', linestyle='-', linewidth=l1, color='#ff7f0e')
# ax.plot(resolution_labels, nearest_accuracy, marker='^', markersize=marker_size, label='Nearest', linestyle='-', linewidth=l1, color='#1f77b4')


# ax.plot(resolution_labels, area_accuracy, marker='s', markersize=marker_size, label='Area', linestyle='--', linewidth=l1)
# ax.plot(resolution_labels, nearest_accuracy, marker='^', markersize=marker_size, label='Nearest', linestyle='-', linewidth=l1)
# ax.plot(resolution_labels, bilinear_accuracy, marker='o', markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1)
# ax.plot(resolution_labels, bicubic_accuracy, marker='*', markersize=marker_size, label='Bicubic', linestyle='-', linewidth=l1)

#
# ax.plot(resolution_labels, area_accuracy, marker=None, markersize=marker_size, label='Area', linestyle='--', linewidth=l1,color='#1f77b4')
# ax.plot(resolution_labels, nearest_accuracy, marker=None, markersize=marker_size, label='Nearest', linestyle='-', linewidth=l1, color='#ff7f0e')
# ax.plot(resolution_labels, bilinear_accuracy, marker=None, markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1, color='#ff7f0e')
# ax.plot(resolution_labels, bicubic_accuracy, marker=None, markersize=marker_size, label='Bicubic', linestyle='-', linewidth=l1,color='#1f77b4')


ax.plot(resolution_labels, area_accuracy, marker="o", markersize=marker_size, label='Area', linestyle='--', linewidth=l1,color='#1f77b4')
ax.plot(resolution_labels, nearest_accuracy, marker="*", markersize=marker_size, label='Nearest', linestyle='--', linewidth=l1,color='#2ca02c')
ax.plot(resolution_labels, bilinear_accuracy, marker="s", markersize=marker_size, label='Bilinear', linestyle='--', linewidth=l1,color='#ff7f0e')
# ax.plot(resolution_labels, bicubic_accuracy, marker=None, markersize=marker_size, label='Bicubic', linestyle='-', linewidth=l1, color='gray')

# ax.scatter(resolution_labels, nearest_accuracy, marker='o', s=marker_size**2, label='Nearest', color='#ff7f0e')

# ax.scatter(resolution_labels, area_accuracy, marker='o', s=marker_size**2, label='Area')
# ax.scatter(resolution_labels, nearest_accuracy, marker='o', s=marker_size**2, label='Nearest')
# ax.scatter(resolution_labels, bilinear_accuracy, marker='o', s=marker_size**2, label='Bilinear')
# ax.scatter(resolution_labels, bicubic_accuracy, marker='o', s=marker_size**2, label='Bicubic',color='gray')



ax.set_xlabel(f'test resolution $r_i$', fontsize=title_fontsize)
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_title(r'Impact of Resize Method', fontsize=title_fontsize)

ax.set_xticks(resolution_labels[::2])
ax.set_xticklabels(resolution_labels[::2])
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)
ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='x', which='minor', length=minor_tick_length)
ax.tick_params(axis='both', which='major', width=line_width, length=minor_tick_length)

ax.legend(fontsize=legend_fontsize)

ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.75)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5)

ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

plt.tight_layout()
plt.show()

fig.savefig('ab-resize.pdf')
