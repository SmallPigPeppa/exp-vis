import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    mean_values = np.array([0.00121648, -0.1486167, 0.03410815, 0.00154123, -0.10826569,
                            0.0225739, -0.03975659, 0.00094656, -0.00114648, -0.03355038,
                            -0.00874648, -0.07328865, -0.04987563, -0.00066305, -0.02232039,
                            -0.0012459, 0.01796051, -0.01137707, -0.03474317, -0.01314376,
                            -0.00098718, -0.05576704, -0.04429319, -0.00395706, -0.0841095,
                            -0.04115568, -0.01216946, 0.0020088, -0.04623547, -0.03166521,
                            -0.01240213, -0.05040688, -0.05743694, -0.00976476, -0.0598396,
                            -0.05033048, -0.01299621, -0.0670504, -0.06056911, -0.01665885,
                            -0.05622633, -0.05753474, -0.02474646, -0.07077304, -0.05933768,
                            -0.01109559, -0.00301553, -0.28401694, -0.09000859, -0.00157768,
                            -0.34196797, -0.0868853, -0.00564744])[1::3]
    var_values = np.array([2.6527023e+00, 5.4060567e-02, 3.1812280e-02, 6.3323546e-03, 3.8960725e-02,
                           2.4252107e-02, 1.2729576e-02, 1.3933826e-03, 1.8808566e-02, 1.8148892e-02,
                           1.0298998e-03, 2.4810685e-02, 3.0001018e-02, 2.3645449e-03, 1.2600677e-02,
                           4.5456484e-02, 2.1591371e-02, 1.6200125e-03, 2.8473567e-02, 2.1454986e-02,
                           1.4215745e-03, 2.4066448e-02, 1.5357634e-02, 9.9170685e-04, 2.9999075e-02,
                           3.1682372e-02, 3.3408867e-03, 8.5237045e-03, 2.1719489e-02, 2.2320516e-02,
                           1.4247188e-03, 1.8276341e-02, 1.5368643e-02, 9.7041950e-04, 1.6341068e-02,
                           1.0054742e-02, 7.4541435e-04, 1.5063213e-02, 9.0366611e-03, 6.9432519e-04,
                           1.4252217e-02, 1.1854451e-02, 1.2736355e-03, 1.3850168e-02, 1.3513955e-02,
                           1.3862501e-03, 1.6742169e-03, 1.4818665e-01, 1.2698019e-02, 8.1761833e-04,
                           2.7458161e-01, 1.0883985e-02, 9.5933105e-04])[1::3]

    means_32 = np.array([0.00796933, -0.12417771, 0.03152151, 0.00151372, -0.09528466,
                         0.01671866, -0.03793498, 0.00087649, -0.0052285, -0.03347373,
                         -0.00915714, -0.07062616, -0.05090125, -0.00061273, -0.02159058,
                         -0.00135048, 0.01796614, -0.01150986, -0.03539963, -0.01335685,
                         -0.00092291, -0.05562486, -0.04463764, -0.00415918, -0.08251248,
                         -0.04163731, -0.01217243, 0.00208454, -0.04606999, -0.03119834,
                         -0.01249147, -0.04979964, -0.05674703, -0.0098832, -0.05845448,
                         -0.04938865, -0.01328387, -0.06379974, -0.06005948, -0.01661204,
                         -0.0521897, -0.05700248, -0.02441496, -0.06311303, -0.06072399,
                         -0.01160182, -0.00211517, -0.28306162, -0.09009556, -0.00175342,
                         -0.3349079, -0.08269363, -0.00553096])[1::3]
    vars_32 = np.array([2.2107420e+00, 3.2392189e-02, 3.3347495e-02, 6.3631954e-03, 3.1653136e-02,
                        2.1376517e-02, 1.3890691e-02, 1.4334349e-03, 1.8370885e-02, 1.7020971e-02,
                        9.3896571e-04, 2.5631633e-02, 2.5091521e-02, 2.2696126e-03, 1.2491344e-02,
                        4.5884497e-02, 2.1937449e-02, 1.6634115e-03, 2.5899010e-02, 2.1785393e-02,
                        1.5263695e-03, 2.2916429e-02, 1.6110027e-02, 9.4667519e-04, 2.8627526e-02,
                        2.4164511e-02, 3.3815391e-03, 7.8314617e-03, 1.9843288e-02, 2.0968659e-02,
                        1.5146681e-03, 1.6251829e-02, 1.3705389e-02, 9.9474005e-04, 1.3298377e-02,
                        8.1922244e-03, 7.7231770e-04, 1.1441864e-02, 7.2630034e-03, 6.9311919e-04,
                        1.0549227e-02, 9.2469640e-03, 1.4519186e-03, 1.0345350e-02, 1.0580670e-02,
                        1.4006461e-03, 1.3118449e-03, 1.2416329e-01, 8.6744763e-03, 8.4612623e-04,
                        2.3553450e-01, 7.2480729e-03, 9.5683854e-04])[1::3]

    x = list(range(len(mean_values)))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    # 在第一个子图上画 mean
    ax1.plot(x, np.abs(mean_values-means_32), '-o', label='Baseline', color='purple', linewidth=1)
    # ax1.plot(x, means_32, label='Baseline 32', color='purple', linewidth=1, linestyle='--', marker='o',
    #          markersize=7, markerfacecolor='none', markeredgewidth=1)
    # ax1.set_xlabel('BN Index')
    ax1.set_ylabel('Mean')
    ax1.set_title('ResNet50', loc='left')
    # ax1.legend()
    ax1.grid()

    # 在第二个子图上画 var
    l1, = ax2.plot(x, np.abs(var_values-vars_32), '-o', label='Baseline', color='purple', linewidth=1)
    # l2, = ax2.plot(x, vars_32, label='Baseline 32', color='purple', linewidth=1, linestyle='--', marker='o',
    #                markersize=7,
    #                markerfacecolor='none', markeredgewidth=1)
    ax2.set_xlabel('BN Index')
    ax2.set_ylabel('Var')
    # ax2.set_title('Var of ResNet50', loc='left')
    # ax2.legend()
    ax2.grid()

    # 显示图形
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.tight_layout()
    plt.savefig('resnet50.pdf', format='pdf', bbox_inches='tight')
    plt.show()