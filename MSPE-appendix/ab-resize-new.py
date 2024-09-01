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

resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 336, 448]
resolution_labels = [str(res) for res in resolutions]

# area_accuracy = [53.068, 59.942, 75.808, 74.16, 78.312, 78.342, 81.188, 79.948, 81.02, 82.558, 85.1, 81.962, 82.104, 74.248]
# bilinear_accuracy = [51.94, 38.838, 77.022, 67.09, 79.022, 66.692, 82.312, 72.232, 76.38, 83.522, 85.1, 79.81, 82.946, 76.584]
pi_resize_accuracy = [0.0145, 0.0325, 0.0578, 0.0903, 0.1301, 0.1770, 0.2312, 0.2926, 0.3613, 0.5202, 0.9248, 2.0809, 3.6994
                      ]
nearest_accuracy = [0.0166, 0.0325, 0.0664, 0.0903, 0.1493, 0.1770, 0.2654, 0.2926, 0.4147, 0.5972, 1.0617, 2.3888,
                    4.2467]
# bicubic_accuracy = [51.356, 36.948, 77.096, 65.33, 78.672, 61.934, 82.386, 64.954, 73.68, 83.784, 84.158, 77.478, 83.024, 77.446]

fig, ax = plt.subplots(figsize=(6 / 0.7, 5.2))

ax.plot(resolution_labels, pi_resize_accuracy, marker="*", markersize=marker_size, label='Non-overlap', linestyle='-',
        linewidth=l1, color='#1f77b4')

ax.plot(resolution_labels, nearest_accuracy, marker="*", markersize=marker_size, label='Overlap', linestyle='--',
        linewidth=l1, color='#ff7f0e')
#
# ax.plot(resolution_labels, bicubic_accuracy, marker="*", markersize=marker_size, label='Bicubic', linestyle='--', linewidth=l1, color='#ff7f0e')


ax.set_xlabel(f'test resolution $r_i$', fontsize=title_fontsize)
ax.set_ylabel('FLOPs (G)', fontsize=title_fontsize)
ax.set_title(r'Computation overhead of MSPE', fontsize=title_fontsize)

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

ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

plt.tight_layout()
plt.show()

fig.savefig('ab-resize.pdf')
