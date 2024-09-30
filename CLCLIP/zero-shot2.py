import matplotlib.pyplot as plt

# 定义任务ID
tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 定义各方法的性能数据 (用你最新的数据更新)
methods = {
    'EWC': [66.87, 58.85, 56.88, 48.08, 47.73, 49.58, 48.63, 46.93, 46.79],
    'ZSCL': [66.87, 60.10, 61.06, 52.28, 49.37, 49.92, 50.65, 52.02, 50.19],
    'Mod-X': [66.87, 59.91, 58.46, 51.50, 45.04, 46.43, 45.82, 48.49, 47.74],
    'MOE-CL': [66.87, 60.06, 61.49, 50.13, 47.29, 48.42, 47.05, 50.71, 49.57],
    'C-CLIP': [66.87, 63.17, 64.78, 64.08, 61.02, 60.55, 62.85, 62.02, 61.58]
}

# 创建绘图
plt.figure(figsize=(10, 6))

# 为每个方法绘制折线
for method, performance in methods.items():
    plt.plot(tasks, performance, marker='o', label=method)

# 设置图表标题和轴标签
plt.title('Zero-shot Performance Across Tasks')
plt.xlabel('Task')
plt.ylabel('Zero-shot Accuracy (%)')

# 显示网格和图例
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
