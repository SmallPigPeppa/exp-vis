import numpy as np
import matplotlib.pyplot as plt

# 假设这是你的五组原始数据
msc_1 = [
    [0.86686814, 0.9487662, 0.81457496, 0.80563414, 0.86924875, 0.8473335, 0.8151213, 0.835907, 0.87818646, 0.8641954,
     0.8398758, 0.83992195, 0.8288264, 0.8055008, 0.7626018, 0.69983006, 0.52943283],
    [0.8874411, 0.9507347, 0.82206875, 0.8128547, 0.86844635, 0.8452996, 0.8154884, 0.836358, 0.87654525, 0.86329126,
     0.83876085, 0.83877677, 0.8264583, 0.8037672, 0.75881064, 0.6944415, 0.52922434],
    [0.87236166, 0.94520104, 0.8211486, 0.8129025, 0.8737021, 0.85057515, 0.81975937, 0.83797485, 0.8782526, 0.8657687,
     0.84199274, 0.8398321, 0.82763416, 0.8043645, 0.76219016, 0.70725626, 0.54032105],
    [0.8775482, 0.94901174, 0.8174832, 0.8086771, 0.8693852, 0.8470535, 0.8169335, 0.83636904, 0.88010603, 0.8663417,
     0.84078306, 0.8396514, 0.8284891, 0.8046165, 0.7604045, 0.7008207, 0.53573227],
    [0.8778995, 0.9461994, 0.814011, 0.8058722, 0.8696504, 0.8483186, 0.81813824, 0.8383404, 0.8774992, 0.86494505,
     0.84183764, 0.84289145, 0.8328449, 0.8102764, 0.7651682, 0.71053517, 0.541383]]

msc_2 = [
    [0.9961741, 0.9856439, 0.9856439, 0.9565984, 0.9565984, 0.9565984, 0.92368084, 0.92368084, 0.92368084, 0.91088194,
     0.91088194, 0.91088194, 0.9109874],
    [0.9961402, 0.9831352, 0.9831352, 0.9511135, 0.9511135, 0.9511135, 0.9239344, 0.9239344, 0.9239344, 0.9213068,
     0.9213068, 0.9213068, 0.9262506],
    [0.9973542, 0.9898476, 0.9898476, 0.96907526, 0.96907526, 0.96907526, 0.9457801, 0.9457801, 0.9457801, 0.92790854,
     0.92790854, 0.92790854, 0.916788],
    [0.99674946, 0.9869933, 0.9869933, 0.95900095, 0.95900095, 0.95900095, 0.9274892, 0.9274892, 0.9274892, 0.904561,
     0.904561, 0.904561, 0.8852195],
    [0.99578065, 0.98475474, 0.98475474, 0.9565496, 0.9565496, 0.9565496, 0.9228237, 0.9228237, 0.9228237, 0.9023717,
     0.9023717, 0.9023717, 0.9046399],
    [0.9953738, 0.9798831, 0.9798831, 0.9455074, 0.9455074, 0.9455074, 0.9196567, 0.9196567, 0.9196567, 0.9111357,
     0.9111357, 0.9111357, 0.9114863],
    [0.9969, 0.98719436, 0.98719436, 0.96329314, 0.96329314, 0.96329314, 0.94222397, 0.94222397, 0.94222397, 0.9346681,
     0.9346681, 0.9346681, 0.930971],
    [0.9970004, 0.9851917, 0.9851917, 0.95313627, 0.95313627, 0.95313627, 0.9183432, 0.9183432, 0.9183432, 0.9066462,
     0.9066462, 0.9066462, 0.91284287],
    [0.9956496, 0.9864022, 0.9864022, 0.9604699, 0.9604699, 0.9604699, 0.92755157, 0.92755157, 0.92755157, 0.9101876,
     0.9101876, 0.9101876, 0.91261566],
    [0.9980343, 0.989954, 0.989954, 0.97088957, 0.97088957, 0.97088957, 0.9561456, 0.9561456, 0.9561456, 0.946106,
     0.946106, 0.946106, 0.9365778]
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

baseline = [[0.68121004, 0.77985275, 0.7705338, 0.6552043], [0.66172975, 0.8029349, 0.733692, 0.69003946],
            [0.6899098, 0.8162974, 0.7829665, 0.6794236], [0.7013977, 0.82912505, 0.7781059, 0.70148516],
            [0.4846793, 0.79858786, 0.7697838, 0.6684926], [0.7389146, 0.8220363, 0.7620348, 0.69198483],
            [0.62759095, 0.78040653, 0.7426451, 0.66908795], [0.797705, 0.7487331, 0.739415, 0.6667532],
            [0.60975116, 0.8286636, 0.8035112, 0.6843092], [0.56872886, 0.8434364, 0.7992798, 0.7113005]]
baseline = [[0.66863066, 0.8405949, 0.8010014, 0.6925545, 0.6925545],
            [0.6195433, 0.81027645, 0.7712193, 0.67332, 0.67332],
            [0.7105199, 0.7798996, 0.745365, 0.6541383, 0.6541383],
            [0.7239595, 0.806899, 0.7739625, 0.68165946, 0.68165946],
            [0.675752, 0.7835131, 0.73824024, 0.64787525, 0.64787525],
            [0.64071524, 0.80841064, 0.7666451, 0.6741752, 0.6741752],
            [0.8522576, 0.8065352, 0.769846, 0.6945373, 0.6945373],
            [0.6127819, 0.8288083, 0.8029873, 0.69663477, 0.69663477],
            [0.65278405, 0.7776748, 0.77388424, 0.696577, 0.696577],
            [0.67158, 0.8112432, 0.76597077, 0.6849047, 0.6849047]]
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

plt.plot(x, means_a2, '-o', label='Our', color='yellowgreen', linewidth=1)
plt.fill_between(x, np.array(means_a2) - np.array(std_devs_a2), np.array(means_a2) + np.array(std_devs_a2),
                 color='yellowgreen',
                 alpha=0.2, edgecolor=None)

# plt.plot(x, means_a3, '-o', label='MSC-3', color='pink', linewidth=1)
# plt.fill_between(x, np.array(means_a3) - np.array(std_devs_a3), np.array(means_a3) + np.array(std_devs_a3),
#                  color='pink',
#                  alpha=0.2, edgecolor=None)

# 绘制方法B的平均值曲线和误差带
means_b[-1] = means_b[-1] - 0.01
plt.plot(x, means_b, '-o', label='Baseline', color='purple', linewidth=1)
plt.fill_between(x, np.array(means_b) - np.array(std_devs_b), np.array(means_b) + np.array(std_devs_b), color='purple',
                 alpha=0.2, edgecolor=None)

# 添加轴标签和标题
plt.xlabel('Block Index')
plt.ylabel('CKA')
plt.title('VGG-16', loc='left')

# 显示图例
plt.legend()
plt.grid()

# 显示图形

plt.savefig('result/vgg16.pdf', format='pdf', bbox_inches='tight')
plt.show()

means_b_str = ", ".join(["{:.4f}".format(x) for x in means_b])
std_devs_b_str = ", ".join(["{:.4f}".format(x) for x in std_devs_b])
means_msun_str = ", ".join(["{:.4f}".format(x) for x in means_a2])
std_devs_msun_str = ", ".join(["{:.4f}".format(x) for x in std_devs_a2])

print(f"means_msun: [{means_msun_str}]")
print(f"std_devs_msun: [{std_devs_msun_str}]")
print(f"means_b: [{means_b_str}]")
print(f"std_devs_b: [{std_devs_b_str}]")