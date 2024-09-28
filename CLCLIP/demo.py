import numpy as np
import matplotlib.pyplot as plt

# 数据
datasets = ['flickr30k', 'coco', 'pet', 'lexica', 'simpsons', 'wikiart', 'kream', 'sketch']
results = {
    'fine-tune': [74.20, 43.68, 16.41, 62.32, 28.50, 53.77, 51.65, 9.95],
    'task0': [35.80, 10.40, 2.02, 3.99, 2.61, 4.62, 9.73, 1.10],
    'task1-flickr30k-our': [80.90, 36.86, 9.46, 22.91, 5.34, 14.10, 13.90, 1.10],
    'task1-flickr30k-ZSCL': [70.55, 21.42, 11.40, 18.00, 4.00, 11.00, 10.00, 1.20],
    'task1-flickr30k-Mod-X': [72.77, 24.19, 12.00, 19.50, 4.50, 12.00, 11.00, 1.30],
    'task1-flickr30k-MOE-CL': [71.21, 22.21, 10.80, 17.20, 4.20, 11.50, 10.50, 1.25],
    'task1-flickr30k-EWC': [72.93, 20.92, 11.00, 18.00, 4.10, 12.00, 10.00, 1.15],
    'task1-flickr30k-POD': [69.49, 20.25, 10.00, 16.50, 3.80, 10.50, 9.50, 1.05]
}

# 绘制
fig, ax = plt.subplots(figsize=(10, 6))

for key, values in results.items():
    ax.plot(datasets, values, marker='o', label=key)

ax.set_xlabel('Datasets')
ax.set_ylabel('Results')
ax.set_title('Performance on Different Datasets')
ax.legend()
ax.grid()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
