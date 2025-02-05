import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

S1 = {}
S2 = {}
S3 = {}
S4 = {}
S2_SNR = np.array([18, 12, 6, 0, -6, -12, -18])
###################################### path to ckpts
# bz=500
# dims: [256], [256], snr: None, None snrNone_None_epoch14
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snrNone_None_epoch14.pth'
# dims: [256], [256], snr: 18, 18
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr18_18_epoch13.pth'
# dims: [256], [256], snr: 12, 12
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr12_12_epoch13.pth'
# dims: [256], [256], snr: 6, 6
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr6_6_epoch13.pth'
# dims: [256], [256], snr: 0, 0
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr0_0_epoch13.pth'
# dims: [256], [256], snr: -6, -6
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr-6_-6_epoch13.pth'
# dims: [256], [256], snr: -12, -12
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr-12_-12_epoch13.pth'
# dims: [256], [256], snr: -18, -18
load_ckpt_path = '.../flickr_ckpt/ckpt/256_256/snr-18_-18_epoch13.pth'



###################################### S1: train without noise, test with noise
##################### testing snr  18     12     6      0     -6     -12     -18
S1['bz500_imgr_r1'] = np.array([67.63, 65.26, 56.03, 28.97, 3.56, 0.27, 0.08])
S1['bz500_imgr_r5'] = np.array([89.82, 88.58, 83.47, 58.17, 11.65, 1.61, 0.52])
S1['bz500_imgr_r10'] = np.array([94.35, 93.76, 90.44, 70.09, 18.33, 2.91, 1.15])
S1['bz500_capr_r1'] = np.array([82.18, 79.06, 66.64, 32.16, 3.84, 0.34, 0.06])
S1['bz500_capr_r5'] = np.array([95.58, 93.94, 87.20, 54.56, 8.38, 0.90, 0.14])
S1['bz500_capr_r10'] = np.array([98.04, 97.38, 92.82, 66.32, 12.54, 1.46, 0.30])

####################################### S2: train with noise, test with noise
# S2_bz100_imgr_r1[training snr][testing snr]
# out_feature_dims = [[256], [256]],
##################### testing snr  18     12     6      0     -6   -12   -18
S2['bz500_imgr_r1'] = np.array([[67.52, 64.99, 55.89, 28.60, 3.60, 0.26, 0.08],  # training snr  18
                                [67.83, 65.82, 57.80, 32.15, 4.53, 0.43, 0.21],  # training snr  12
                                [67.74, 66.04, 58.55, 32.34, 3.64, 0.25, 0.12],  # training snr  6
                                [60.36, 59.20, 54.31, 37.16, 7.59, 0.61, 0.18],  # training snr  0
                                [32.55, 31.96, 29.83, 23.05, 10.72, 1.66, 0.40],  # training snr  -6
                                [6.21, 6.17, 6.01, 5.45, 3.85, 1.72, 0.46],  # training snr  -12
                                [0.21, 0.21, 0.21, 0.22, 0.21, 0.17, 0.22]])  # training snr  -18
S2['bz500_imgr_r5'] = np.array([[89.80, 88.58, 83.46, 57.95, 11.58, 1.64, 0.50],
                                [89.75, 88.91, 84.63, 62.04, 13.80, 1.61, 0.73],
                                [89.86, 89.08, 85.33, 63.18, 12.55, 1.40, 0.53],
                                [85.78, 85.28, 82.71, 68.87, 23.07, 2.76, 0.86],
                                [64.95, 64.42, 62.46, 54.20, 31.01, 7.12, 1.30],
                                [20.42, 20.40, 19.97, 18.84, 14.41, 6.74, 1.86],
                                [1.16, 1.20, 1.16, 1.15, 1.08, 1.05, 1.08]])
S2['bz500_imgr_r10'] =np.array([[94.34, 93.74, 90.41, 70.27, 18.02, 2.87, 1.14],
                                [94.38, 93.85, 91.38, 73.75, 21.17, 3.14, 1.30],
                                [94.25, 93.89, 91.56, 75.11, 19.39, 2.64, 1.10],
                                [92.14, 91.79, 90.12, 79.93, 32.76, 4.77, 1.67],
                                [77.16, 76.80, 74.97, 68.33, 44.16, 11.78, 2.38],
                                [31.82, 31.81, 31.42, 29.50, 23.48, 11.90, 3.73],
                                [2.40, 2.45, 2.40, 2.28, 2.26, 2.04, 1.91]])
S2['bz500_capr_r1'] =np.array([[81.84, 78.84, 66.94, 32.88, 3.94, 0.28, 0.08],
                               [82.64, 79.54, 69.68, 37.38, 4.40, 0.46, 0.20],
                               [83.68, 81.24, 70.40, 36.88, 4.08, 0.24, 0.10],
                               [77.84, 75.34, 67.24, 43.28, 7.92, 0.48, 0.16],
                               [39.62, 38.86, 35.82, 26.58, 11.84, 1.68, 0.26],
                               [6.64, 6.74, 6.60, 5.90, 4.20, 1.50, 0.54],
                               [0.28, 0.30, 0.28, 0.26, 0.24, 0.12, 0.32]])
S2['bz500_capr_r5'] =np.array([[95.60, 94.04, 87.40, 54.92, 8.58, 0.90, 0.12],
                               [95.88, 94.84, 89.72, 60.22, 9.98, 0.90, 0.38],
                               [96.44, 95.60, 90.30, 60.10, 8.86, 0.60, 0.16],
                               [94.26, 93.12, 89.18, 68.06, 16.58, 1.22, 0.30],
                               [70.96, 70.14, 66.48, 53.78, 25.34, 4.00, 0.60],
                               [19.24, 19.04, 18.48, 16.40, 10.78, 4.14, 0.96],
                               [0.98, 0.96, 0.98, 0.94, 0.84, 0.74, 0.54]])
S2['bz500_capr_r10'] =np.array([[98.16, 97.38, 93.14, 66.82, 12.42, 1.30, 0.32],
                                [98.12, 97.66, 94.60, 71.72, 15.06, 1.54, 0.54],
                                [98.52, 98.34, 95.72, 71.56, 12.82, 1.10, 0.34],
                                [97.10, 96.70, 94.56, 79.82, 24.26, 2.18, 0.54],
                                [82.66, 82.12, 78.52, 67.50, 35.40, 6.48, 1.10],
                                [28.52, 28.50, 27.80, 24.94, 16.86, 7.24, 1.58],
                                [1.78, 1.72, 1.72, 1.58, 1.50, 1.24, 0.98]])


###################################### S3: train without noise, test without noise
S3['bz500_imgr_r1'] = 68.48
S3['bz500_imgr_r5'] = 90.28
S3['bz500_imgr_r10'] = 94.52
S3['bz500_capr_r1'] = 83.40
S3['bz500_capr_r5'] = 96.20
S3['bz500_capr_r10'] = 98.40

###################################### S4: train with noise, test without noise
################## training snr  18     12     6      0     -6    -12    -18
S4['bz500_imgr_r1'] =np.array([68.22, 68.42, 68.46, 60.66, 32.58, 6.24, 0.22])
S4['bz500_imgr_r5'] =np.array([90.16, 89.86, 90.08, 85.90, 65.18, 20.56, 1.18])
S4['bz500_imgr_r10'] =np.array([94.58, 94.60, 94.44, 92.28, 77.20, 31.68, 2.36])
S4['bz500_capr_r1'] =np.array([83.20, 83.80, 85.10, 78.50, 39.90, 6.70, 0.30])
S4['bz500_capr_r5'] =np.array([96.50, 96.90, 97.00, 94.80, 71.20, 19.20, 1.00])
S4['bz500_capr_r10'] =np.array([98.20, 98.20, 98.50, 97.00, 82.60, 29.10, 1.80])

################################################################# plot ###################################################################


# #######  S2 compare batch sizes
# X, Y = np.meshgrid(S2_SNR, S2_SNR) # X:testing, Y: Training
# fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
# fig.set_size_inches(16, 9)
# for j, ic in enumerate(['img', 'cap']):
#      for i, r in enumerate([1, 5, 10]):
#          ax[j, i].plot_wireframe(X, Y, S2[f'bz500_{ic}r_r{r}'], color='b')
#          ax[j, i].set_xlabel('Test SNR/dB', fontsize=12)
#          ax[j, i].set_ylabel('Train SNR/dB', fontsize=12)
#          ax[j, i].set_zlabel('Accuracy/%', fontsize=15)
#          ax[j, i].set_zlim(0, 100)
#          ax[j, i].view_init(20, 300, None)
# ax[0, 0].annotate('R@1', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[0, 1].annotate('R@5', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[0, 2].annotate('R@10', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[0, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[0, 0].annotate('Image', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[1, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# ax[1, 0].annotate('Caption', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# plt.subplots_adjust(wspace=0, hspace=0)
# lines_labels = [ax[0, 0].get_legend_handles_labels()]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels, loc='upper left', fontsize='xx-large')
# plt.show()

#######  S2 find best training snr for given testing snr
# x: testing snr, y: best training snr
fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(16, 9)
for j, ic in enumerate(['img', 'cap']):
     for i, r in enumerate([1, 5, 10]):
         ax[j, i].plot(S2_SNR, S2_SNR[np.argmax(S2[f'bz500_{ic}r_r{r}'], axis=0)], 'b--', linewidth=3.0, label='best train SNR for given test SNR')
         ax[j, i].plot(S2_SNR, S2_SNR, 'r:', linewidth=3.0, label='train SNR=test SNR')
         ax[j, i].set_xlabel('Given test snr', fontsize=15)
         ax[j, i].set_ylabel('Best train snr', fontsize=15)
ax[0, 0].annotate('R@1', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 1].annotate('R@5', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 2].annotate('R@10', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('retrieval', xy=(-0.35, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('Image', xy=(-0.35, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('retrieval', xy=(-0.35, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('Caption', xy=(-0.35, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# plt.subplots_adjust(wspace=0, hspace=0)
lines_labels = [ax[0, 0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper left', fontsize='xx-large')
plt.show()



#######  all 4 Scenarios, bz500
fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(16, 9)
for j, ic in enumerate(['img', 'cap']):
     for i, r in enumerate([1, 5, 10]):
         ax[j, i].plot(S2_SNR, np.array([S3[f'bz500_{ic}r_r{r}']]*S2_SNR.shape[0]), color='r', linewidth=3.0, label='Scenario A')
         ax[j, i].plot(S2_SNR, S1[f'bz500_{ic}r_r{r}'], 'b--*', linewidth=3.0, markersize=12, label='Scenario B')
         ax[j, i].plot(S2_SNR, S4[f'bz500_{ic}r_r{r}'], 'b--o', linewidth=3.0, markersize=10, label='Scenario C')
         ax[j, i].plot(S2_SNR, np.diag(S2[f'bz500_{ic}r_r{r}']), color='b', linewidth=3.0, label='Scenario D')
         ax[j, i].set_xlabel('SNR/dB', fontsize=15)
         ax[j, i].set_ylabel('Accuracy/%', fontsize=15)
         ax[j, i].set_ylim(0, 100)
ax[0, 0].annotate('R@1', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 1].annotate('R@5', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 2].annotate('R@10', xy=(0.5, 1.1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('retrieval', xy=(-0.35, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('Image', xy=(-0.35, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('retrieval', xy=(-0.35, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('Caption', xy=(-0.35, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
# plt.subplots_adjust(wspace=0, hspace=0)
lines_labels = [ax[0, 0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper left', fontsize='xx-large')
plt.show()

