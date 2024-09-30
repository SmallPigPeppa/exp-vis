import matplotlib.pyplot as plt

# 定义任务ID
tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 定义各方法的性能数据
methods = {
    'EWC': [67.73, 59.85, 46.67, 38.43, 45.27, 42.00, 39.10, 40.50, 38.76],
    'ZSCL': [67.73, 64.70, 59.51, 52.47, 54.74, 52.03, 52.89, 51.02, 48.60],
    'Mod-X': [67.73, 61.63, 53.05, 45.15, 48.53, 45.79, 44.76, 42.63, 42.81],
    'MOE-CL': [67.73, 62.92, 55.29, 51.08, 52.75, 50.33, 47.61, 46.88, 45.05],
    'C-CLIP': [67.73, 63.11, 65.31, 63.26, 63.31, 61.95, 62.13, 61.51, 60.31]
}

# 创建绘图
plt.figure(figsize=(10, 6))

# 为每个方法绘制折线
for method, performance in methods.items():
    plt.plot(tasks, performance, marker='o', label=method)

# 设置图表标题和轴标签
plt.title('xxx')
plt.xlabel('task')
plt.ylabel('zero-shot')

# 显示网格和图例
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
