import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


def custom_formatter(x, pos):
    return f'{int(x)}%'


title_fontsize = 33
tick_label_fontsize = 26
legend_fontsize = 28
line_width = 3  # Adjust line width
minor_tick_length = 8
l1 = 1.5
marker_size = 7

# Modified Resolutions and Corresponding Accuracies
resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 448, 896]
resolution_labels = [str(res) for res in resolutions]

# Filtered Accuracies
vanilla_accuracy = [37.23, 57.58, 67.28, 72.25, 75.41, 77.45, 78.76, 80.43, 81.35, 81.75, 82.18, 81.86, 81.12]
flexivit_accuracy = [38.94, 62.07, 70.23, 74.20, 76.76, 78.90, 80.26, 81.60, 82.22, 82.53, 82.73, 82.54, 82.53]
mspe_accuracy = [67.99, 75.17, 79.53, 81.65, 82.65, 83.13, 83.75, 83.71, 83.92, 83.94, 83.89, 83.97, 83.97]

# Plotting accuracies with modified resolutions
fig, ax = plt.subplots(figsize=(6 / 0.7, 6))

# Plotting accuracies
ax.plot(resolution_labels, vanilla_accuracy, label='Vanilla', linestyle='--', linewidth=l1, marker='*', markersize=marker_size)
ax.plot(resolution_labels, flexivit_accuracy, label='FlexiViT', linestyle='--', linewidth=l1, marker='^',
        markersize=marker_size)
# ax.plot(resolution_labels, vanilla_accuracy, label='Vanilla', linestyle='--', linewidth=l1, marker='o', markersize=marker_size)
# ax.plot(resolution_labels, flexivit_accuracy, label='FlexiViT', linestyle='--', linewidth=l1, marker='s',
#         markersize=marker_size)
ax.plot(resolution_labels, mspe_accuracy, marker=None, label='MSPE', linewidth=l1, color='r')

# Set titles and labels
ax.set_xlabel(f'test width $w_i$', fontsize=title_fontsize)
ax.set_ylabel('ImageNet-1K Acc@1', fontsize=title_fontsize)
ax.set_title(f'(b) $height=128$', fontsize=title_fontsize)

# Set x-axis tick labels
ax.set_xticks(resolution_labels[::2])
ax.set_xticklabels(resolution_labels[::2])

# Set minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(1))

# Set axis visibility and line width
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(line_width)
ax.spines['left'].set_linewidth(line_width)
ax.tick_params(axis='both', labelsize=tick_label_fontsize)
ax.tick_params(axis='x', which='minor', width=line_width, length=minor_tick_length)
ax.tick_params(axis='both', which='major', width=line_width, length=minor_tick_length)

# Add legend
ax.legend(fontsize=legend_fontsize)

# Display grid
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1.0)
ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.7)

# Set y-axis ticks
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

# Set y-axis formatter
ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

plt.tight_layout()

# Display and save the figure
plt.show()
fig.savefig('intro-fig2.pdf')
