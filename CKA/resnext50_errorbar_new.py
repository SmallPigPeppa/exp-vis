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
    [0.7261079, 0.8637194, 0.81801355, 0.9238493, 0.9389342, 0.84548986, 0.7918449, 0.7493827, 0.792423, 0.806652,
     0.8031177, 0.8199926, 0.83267725, 0.76134574, 0.8211259, 0.75483024, 0.70575166],
    [0.7351726, 0.8683021, 0.8057414, 0.92255723, 0.93918335, 0.8336286, 0.7537488, 0.6871467, 0.7034161, 0.7258237,
     0.7224162, 0.7482665, 0.7713979, 0.6630875, 0.7855953, 0.67515665, 0.7120486],
    [0.7431396, 0.87369597, 0.81105864, 0.9248887, 0.94259775, 0.85254, 0.7976745, 0.7548481, 0.79521894, 0.80871093,
     0.8043484, 0.8222956, 0.83580005, 0.766477, 0.82377595, 0.7552215, 0.7072351],
    [0.7471612, 0.8722119, 0.8182769, 0.9246001, 0.9408595, 0.84376496, 0.77011186, 0.7063004, 0.72281355, 0.74730253,
     0.74463105, 0.7685246, 0.7893276, 0.6904878, 0.79333454, 0.6924499, 0.7090521],
    [0.7241754, 0.8715937, 0.82460153, 0.93005544, 0.947254, 0.8698081, 0.83774763, 0.8144415, 0.86352676, 0.86385375,
     0.85845584, 0.86476403, 0.87092406, 0.83265346, 0.8507407, 0.8098591, 0.70483685]
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
baseline = [[0.69426507, 0.82879037, 0.84728485, 0.8956484, 0.37134457],
            [0.7318271, 0.82034546, 0.847311, 0.8907841, 0.3745986],
            [0.6486183, 0.8035253, 0.8095117, 0.84355074, 0.39280716],
            [0.65475535, 0.825149, 0.8391764, 0.8911507, 0.4239638],
            [0.67755854, 0.8172854, 0.84184426, 0.9011511, 0.3961586],
            [0.66743606, 0.81054676, 0.81525856, 0.8524587, 0.38229328],
            [0.6162845, 0.8182557, 0.8119056, 0.8408636, 0.39599678],
            [0.7330103, 0.8491427, 0.84767365, 0.8783139, 0.3613436],
            [0.6642602, 0.80072993, 0.8270861, 0.8876339, 0.37972575],
            [0.7321905, 0.8324224, 0.8328075, 0.85263675, 0.41428053]]
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
#

# 绘制方法A的平均值曲线和误差带
# plt.plot(x, means_a1, '-o', label='MSC-1', color='blue', linewidth=1)
# plt.fill_between(x, np.array(means_a1) - np.array(std_devs_a1), np.array(means_a1) + np.array(std_devs_a1),
#                  color='blue',
#                  alpha=0.2, edgecolor=None)

# plt.plot(x, means_a2, '-o', label='Our', color='yellowgreen', linewidth=1)
# plt.fill_between(x, np.array(means_a2) - np.array(std_devs_a2), np.array(means_a2) + np.array(std_devs_a2),
#                  color='yellowgreen',
#                  alpha=0.2, edgecolor=None)

# plt.plot(x, means_a3, '-o', label='MSC-3', color='pink', linewidth=1)
# plt.fill_between(x, np.array(means_a3) - np.array(std_devs_a3), np.array(means_a3) + np.array(std_devs_a3),
#                  color='pink',
#                  alpha=0.2, edgecolor=None)

# 绘制方法B的平均值曲线和误差带
# means_b[-1] = means_b[-1] - 0.01
plt.plot(x, means_b, '-o', label='Baseline', color='purple', linewidth=1)
plt.fill_between(x, np.array(means_b) - np.array(std_devs_b), np.array(means_b) + np.array(std_devs_b), color='purple',
                 alpha=0.2, edgecolor=None)

# 添加轴标签和标题
plt.xlabel('Block Index')
plt.ylabel('CKA')
plt.title('VGG16', loc='left')

# 显示图例
plt.legend()
plt.grid()

# 显示图形

plt.savefig('resnext50.pdf', format='pdf', bbox_inches='tight')
plt.show()
