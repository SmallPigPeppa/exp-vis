import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(feats_path):
    feats=np.load(feats_path)
    return cosine_similarity(feats)

if __name__=='__main__':

    # # byol_feats=''
    # # byol_cos=cos_sim(byol_feats)
    # similarity_matrices = [np.random.rand(10, 10) for i in range(5)]
    #
    # # 设置图像大小
    # fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    #
    # # 循环绘制五个图
    # for i, similarity_matrix in enumerate(similarity_matrices):
    #     axs[i].imshow(similarity_matrix)
    #     axs[i].set_xlabel("Sample Index")
    #     axs[i].set_ylabel("Sample Index")
    #
    # # 在最后一个图的右边添加颜色条
    # cbar = fig.colorbar(axs[-1].imshow(similarity_matrices[-1]), ax=axs, orientation="vertical")
    # cbar.set_label("Cosine Similarity")
    #
    #
    #
    #
    # # plt.tight_layout()
    # plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    # 假设有五个余弦相似度矩阵
    matrices = [np.random.rand(10, 10) for i in range(5)]

    # 绘制五个子图
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    for ax, matrix in zip(axs, matrices):
        im = ax.imshow(matrix, cmap='coolwarm')

    # 添加一个颜色条
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()
    plt.show()

    # # sort samplers by label
    # sx = np.load('./s_test_x.npy')
    # sx2=np.load('./s_test0.npy')
    # sy=np.load('./s_test_y.npy')
    # sslx = np.load('./ssl_test_test.npy')
    # ssly = np.load('./y_test0.npy')
    # sslorder = ssly.argsort()
    # sorder = sy.argsort()
    # sslx = sslx[sslorder]
    # sx = sx[sorder]
    # sx2=sx2[sorder]
    # print(ssly)
    # print(sorder)
    # print(ssly[sslorder])
    # num_samplers = 10000
    # a1=np.ones(shape=(10000,10000))
    # a0=np.zeros(shape=(1000,1000))
    # for i in range(10):
    #     a1[i*1000:(i+1)*1000,i*1000:(i+1)*1000]=a0
    #
    #
    # print(ssly.shape)
    # print(sslx.shape)
    # print(sx.shape)
    # s_cos=cosine_similarity(X=sx, dense_output=True)
    # s2_cos=cosine_similarity(X=sx2, dense_output=True)
    # ssl_cos=cosine_similarity(X=sslx, dense_output=True)
    # # ssl_cos=ssl_cos-a1*0.02
    # # s_cos=s_cos+0.003*a1
    # print(s_cos.shape)
    # print(ssl_cos)
    # np.save('./s_cos', s_cos)
    # np.save('./ssl_cos', ssl_cos)
    #
    # fig, ax = plt.subplots()
    # plt.imshow(ssl_cos,cmap=plt.cm.Blues)
    # cbar=plt.colorbar()
    # ax.tick_params(axis='both', which='major', labelsize=13)
    # plt.clim(0, 1)
    # for t in cbar.ax.get_yticklabels():
    #      t.set_fontsize(13)
    # fig.savefig('ssl_cos.png', format='png', dpi=300)
    # plt.show()
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设有5个余弦相似度矩阵，每个矩阵的大小为10x10
    cos_sim_matrices = [np.random.rand(10, 10) for i in range(5)]

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 8))

    for ax, cos_sim_matrix in zip(axs.flat, cos_sim_matrices):
        im = ax.imshow(cos_sim_matrix)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

