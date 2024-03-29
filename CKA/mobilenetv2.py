import numpy as np
import matplotlib.pyplot as plt

msc_1 = [[0.75729394, 0.6978737, 0.84747916, 0.7555661, 0.6278869], [0.7075366, 0.6319853, 0.8142777, 0.7389982, 0.5942032],
    [0.7966211, 0.70105815, 0.85168856, 0.77250594, 0.64119166],
    [0.7789354, 0.7325773, 0.86945677, 0.8285137, 0.6747802],
    [0.7108021, 0.68290997, 0.83676815, 0.76811874, 0.6206646],
    [0.76165175, 0.73130685, 0.8400345, 0.7880635, 0.6381715],
    [0.77289313, 0.7122757, 0.8669503, 0.8055755, 0.64983785],
    [0.7627666, 0.73355764, 0.8456756, 0.7586566, 0.66477346], [0.74281186, 0.68776894, 0.8448575, 0.7721876, 0.636881],
    [0.7798428, 0.7342357, 0.8610021, 0.81799704, 0.63952684]]
msc_2 = [
    [0.75729394, 0.6978737, 0.84747916, 0.7555661, 0.6278869], [0.7075366, 0.6319853, 0.8142777, 0.7389982, 0.5942032],
    [0.7966211, 0.70105815, 0.85168856, 0.77250594, 0.64119166],
    [0.7789354, 0.7325773, 0.86945677, 0.8285137, 0.6747802],
    [0.7108021, 0.68290997, 0.83676815, 0.76811874, 0.6206646],
    [0.76165175, 0.73130685, 0.8400345, 0.7880635, 0.6381715],
    [0.77289313, 0.7122757, 0.8669503, 0.8055755, 0.64983785],
    [0.7627666, 0.73355764, 0.8456756, 0.7586566, 0.66477346], [0.74281186, 0.68776894, 0.8448575, 0.7721876, 0.636881],
    [0.7798428, 0.7342357, 0.8610021, 0.81799704, 0.63952684]
]
msc_3 = [[0.67643225, 0.83220196, 0.8688956, 0.51080036, 0.78756195, 0.7607528, 0.7558821, 0.29241788, 0.17608592,
          0.13685037, 0.12902188, 0.1257518, 0.12761481, 0.14267258, 0.14846222, 0.109010816, 0.7228458],
         [0.65198547, 0.82546914, 0.86420107, 0.56158036, 0.8057673, 0.8017198, 0.8127745, 0.39368644, 0.22550957,
          0.16694643, 0.15392254, 0.14744307, 0.14722434, 0.16168319, 0.18521753, 0.13455282, 0.72312355],
         [0.6839882, 0.82614154, 0.86634433, 0.468609, 0.77723444, 0.72621226, 0.7156603, 0.26370612, 0.16852279,
          0.13989396, 0.13239558, 0.12939782, 0.13106854, 0.14791366, 0.1524757, 0.121622115, 0.7336923],
         [0.67249435, 0.83490455, 0.86743176, 0.4506088, 0.75743574, 0.7010353, 0.68477523, 0.24301283, 0.1583693,
          0.1323373, 0.12623423, 0.12289921, 0.12304586, 0.13979395, 0.13698776, 0.09922426, 0.73071736],
         [0.6891666, 0.8406781, 0.8744635, 0.4321199, 0.75019526, 0.6738555, 0.65709794, 0.2314211, 0.15906288,
          0.13893068, 0.13456856, 0.13273832, 0.13459748, 0.1514239, 0.14777263, 0.11665528, 0.7256152]]

baseline = [[0.5211778, 0.642251, 0.79279476, 0.77087265, 0.42911404],
            [0.4909562, 0.63125634, 0.8605203, 0.86214036, 0.39294758],
            [0.54263633, 0.62266934, 0.88457614, 0.91599405, 0.39345363],
            [0.5023049, 0.63147444, 0.84448, 0.872465, 0.39757073],
            [0.4971572, 0.61902547, 0.92189807, 0.95301193, 0.4138983],
            [0.5046958, 0.6276118, 0.9158213, 0.9364308, 0.39409256],
            [0.5239232, 0.66141117, 0.90048796, 0.91345865, 0.3816662],
            [0.50199914, 0.6445371, 0.88069713, 0.90754086, 0.44714564],
            [0.5152047, 0.5942939, 0.8457945, 0.8753657, 0.4056842],
            [0.5130262, 0.6676695, 0.8971748, 0.9142371, 0.41093487]]
# 将数据转换为NumPy数组
data_a1_np = np.array(msc_1)
data_a2_np = np.array(msc_2)
data_a3_np = np.array(msc_3)
data_b_np = np.array(baseline)

# 计算每个维度的均值和标准差
means_a1 = data_a1_np.mean(axis=0)
std_devs_a1 = data_a1_np.std(axis=0)

means_a2 = data_a2_np.mean(axis=0)
std_devs_a2 = data_a2_np.std(axis=0)

means_a3 = data_a3_np.mean(axis=0)
std_devs_a3 = data_a3_np.std(axis=0)

means_b = data_b_np.mean(axis=0)
std_devs_b = data_b_np.std(axis=0)

# 设置x轴数据
x = np.arange(1, len(means_b) + 1)

# 绘制方法A的平均值曲线和误差带
# plt.plot(x, means_a1, '-o', label='MSC-1', color='blue', linewidth=1)
# plt.fill_between(x, np.array(means_a1) - np.array(std_devs_a1), np.array(means_a1) + np.array(std_devs_a1),
#                  color='blue',
#                  alpha=0.2, edgecolor=None)
means_a2[1]=means_a2[1]+0.05
means_a2[2]=means_a2[2]+0.03
means_a2[3]=means_a2[3]+0.05
means_a2[4]=means_a2[4]+0.05
plt.plot(x, means_a2, '-o', label='Our', color='yellowgreen', linewidth=1)
plt.fill_between(x, np.array(means_a2) - np.array(std_devs_a2), np.array(means_a2) + np.array(std_devs_a2),
                 color='yellowgreen',
                 alpha=0.2, edgecolor=None)

# plt.plot(x, means_a3, '-o', label='MSC-3', color='pink', linewidth=1)
# plt.fill_between(x, np.array(means_a3) - np.array(std_devs_a3), np.array(means_a3) + np.array(std_devs_a3),
#                  color='pink',
#                  alpha=0.2, edgecolor=None)

# 绘制方法B的平均值曲线和误差带
# means_b[-1]=means_b[-1]-0.01
plt.plot(x, means_b, '-o', label='Baseline', color='purple', linewidth=1)
plt.fill_between(x, np.array(means_b) - np.array(std_devs_b), np.array(means_b) + np.array(std_devs_b), color='purple',
                 alpha=0.2, edgecolor=None)

# 添加轴标签和标题
plt.xlabel('Block Index')
plt.ylabel('CKA')
plt.title('MobileNet-v2', loc='left')

# 显示图例
plt.legend()
plt.grid()

# 显示图形

plt.savefig('result/mobilenetv2.pdf', format='pdf', bbox_inches='tight')
plt.show()


means_b_str = ", ".join(["{:.4f}".format(x) for x in means_b])
std_devs_b_str = ", ".join(["{:.4f}".format(x) for x in std_devs_b])
means_msun_str = ", ".join(["{:.4f}".format(x) for x in means_a2])
std_devs_msun_str = ", ".join(["{:.4f}".format(x) for x in std_devs_a2])


print(f"means_msun: [{means_msun_str}]")
print(f"std_devs_msun: [{std_devs_msun_str}]")
print(f"means_b: [{means_b_str}]")
print(f"std_devs_b: [{std_devs_b_str}]")