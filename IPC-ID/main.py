import numpy as np
import torch
import matplotlib.pyplot as plt

# sort samplers by label
sx = np.load('./s_test0.npy')
sslx = np.load('./ssl_test0.npy')
y = np.load('./y_test0.npy')
order = y.argsort()
sx = sx[order]
sslx = sslx[order]
print(y)
print(order)
print(y[order])

# # normalization for every classes
# sx=torch.from_numpy(sx)
# sslx=torch.from_numpy(sslx)

#
s_sv= []
ssl_sv = []
fig, ax = plt.subplots()
components=20
for i in range(10):
    sx_i = sx[100 * i:100 * (i + 1)]
    sslx_i = sslx[100 * i:100 * (i + 1)]
    sx_i = torch.from_numpy(sx_i)
    sslx_i = torch.from_numpy(sslx_i)
    sx_meani = torch.mean(sx_i, dim=0)
    sslx_meani = torch.mean(sslx_i, dim=0)
    sx_i = sx_i - sx_meani
    sslx_i = sslx_i - sslx_meani

    u, s, v = torch.svd(sx_i)
    u2, s2, v2 = torch.svd(sslx_i)
    s = s / torch.max(s)
    s2 = s2 / torch.max(s2)
    if i ==0:
        ax.plot(np.square(s.numpy()[:components]), marker='o', color='g', label='Supervised', linewidth=0.3, markersize=2.0)
        ax.plot(np.square(s2.numpy()[:components]), marker='o', color='r', label='Self-Supervised', linewidth=0.3,markersize=2.0)
    else:
        ax.plot(np.square(s.numpy()[:components]), marker='o', color='g', linewidth=0.3,
                markersize=2.0)
        ax.plot(np.square(s2.numpy()[:components]), marker='o', color='r', linewidth=0.3,
                markersize=2.0)
ax.set_xticks(list(range(components)))
plt.grid()
ax.legend(loc='upper right')
plt.xlabel("Components")
plt.ylabel("Sigular Values")
plt.savefig(f'splot_inner_{components}.pdf')
