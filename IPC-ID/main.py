import numpy as np
import torch
import matplotlib.pyplot as plt

# sort samplers by label
byol_feats = np.load('Pre-Feats/byol_features.npy')
simclr_feats= np.load('Pre-Feats/simclr_features.npy')



x_all = byol_feats
x_all2= simclr_feats
x_all=torch.from_numpy(x_all)
x_all2=torch.from_numpy(x_all2)
x_all_mean=torch.mean(x_all,dim=0)
x_all_mean2=torch.mean(x_all2,dim=0)

x_all=x_all-x_all_mean
x_all2=x_all2-x_all_mean2
u, s, v = torch.svd(x_all)
u2, s2, v2 = torch.svd(x_all2)






s=s/torch.max(s)
s2=s2/torch.max(s2)
print(s)

s=torch.square(s)
s2=torch.square(s2)



#计算有效维
s_sum=torch.sum(s)*0.9
s2_sum=torch.sum(s2)*0.9
print(s)
print(s2)
print(s.shape,s2.shape)
print(s_sum,s2_sum)
s_sum0=0
s2_sum0=0
v1=0
v2=0
while(s_sum0<s_sum):
    s_sum0=s_sum0+s.numpy()[v1]
    v1=v1+1
while(s2_sum0<s2_sum):
    s2_sum0=s2_sum0+s2.numpy()[v2]
    v2=v2+1
print(v1)
print(v2)




fig, ax = plt.subplots()

components=20
# ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0.918, 0.917, 0.945))
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', color='white')
ax.tick_params(axis='y', color='white')
# 'coral' (0.935,0.524,0.357)
ax.text(11.3, 0.65, 'feature space of SL\nPC-ID=9', style='italic',color='coral',ha='center',
        bbox={'facecolor': 'coral', 'alpha': 0.12, 'pad': 10,'linewidth':0 },fontsize=12
        )
ax.text(13.5, 0.4, 'feature space of SSL\nPC-ID=182', style='italic',color='green',ha='center',
        bbox={'facecolor': 'green', 'alpha': 0.12, 'pad': 10,'linewidth':0 },fontsize=12
        )
# ax.text(5, 0.6, 'boxed italics text in data coords', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
# green coral
ax.plot(s.numpy()[:components],marker='o',color='coral',label='Supervised',linewidth=1.5,markersize=7,linestyle='--')
ax.plot(s2.numpy()[:components],marker='o',color='green',label='Self-Supervised',linewidth=1.5,markersize=7,linestyle='--')
ax.plot(range(9,components),s.numpy()[9:components],marker='o',color='coral',linewidth=1.5,markersize=7)
ax.set_xticks(list(range(components)))
plt.grid(color='white')
# ,axis='y'
ax.legend(loc='upper right',fontsize=13,)
# framealpha=0.8 ,edgecolor='black' fancybox=False,  shadow=False, borderpad=0.5,
#           labelspacing=0.3,
# bbox_to_anchor = (0.95, 0.95)
plt.xlabel("Components",fontsize=16)
plt.ylabel("Normalized Eigenvalues",fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(f'splot_{components}.png',dpi=500)
plt.show()
#



# components=100
# ax.plot(np.square(s.numpy()[:components]),marker='o',color='g',label='Supervised',linewidth=0.5,markersize=2.0)
# ax.plot(np.square(s2.numpy()[:components]),marker='o',color='r',label='Self-Supervised',linewidth=0.5,markersize=2.0)
# ax.set_xticks(list(range(0, components+1, 5)))
# plt.grid()
# ax.legend(loc='upper right')
# plt.xlabel("Components")
# plt.ylabel("Normalized Eigenvalues")
# plt.savefig(f'splot_{components}.pdf')
