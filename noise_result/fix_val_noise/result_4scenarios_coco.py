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
load_ckpt_path = '.../coco_ckpt/ckpt_coco/snrNone_None_epoch43.pth'
# dims: [256], [256], snr: 18, 18
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr18_18_epoch36.pth'
# dims: [256], [256], snr: 12, 12
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr12_12_epoch36.pth'
# dims: [256], [256], snr: 6, 6
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr6_6_epoch49.pth'
# dims: [256], [256], snr: 0, 0
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr0_0_epoch49.pth'
# dims: [256], [256], snr: -6, -6
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr-6_-6_epoch47.pth'
# dims: [256], [256], snr: -12, -12
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr-12_-12_epoch34.pth'
# dims: [256], [256], snr: -18, -18
load_ckpt_path = '.../coco_ckpt/ckpt_coco/ckpt_coco/snr-18_-18_epoch24.pth'



###################################### S1: train without noise, test with noise
##################### testing snr  18     12     6      0     -6     -12     -18
S1['bz100_imgr_r1'] = np.array([63.33, 61.59, 55.66, 36.89, 8.05, 0.67, 0.19])
S1['bz100_imgr_r5'] = np.array([89.95, 89.26, 85.91, 70.19, 22.04, 2.86, 0.77])
S1['bz100_imgr_r10'] = np.array([95.82, 95.49, 93.88, 82.98, 31.38, 4.85, 1.79])
S1['bz100_capr_r1'] = np.array([74.72, 71.84, 63.60, 40.06, 8.08, 0.56, 0.12])
S1['bz100_capr_r5'] = np.array([93.56, 91.94, 85.36, 62.98, 15.36, 1.34, 0.24])
S1['bz100_capr_r10'] = np.array([97.02, 95.78, 92.06, 74.40, 22.26, 2.44, 0.44])
####################################### S2: train with noise, test with noise
##################### testing snr  18     12     6      0     -6   -12   -18
S2['bz100_imgr_r1'] = np.array([[63.44, 61.86, 55.69, 34.96, 6.05, 0.58, 0.16],  # training snr  18
                                [63.84, 62.18, 56.21, 36.46, 6.15, 0.36, 0.07],  # training snr  12
                                [63.32, 61.76, 55.82, 36.64, 6.28, 0.39, 0.18],  # training snr  6
                                [55.68, 54.35, 50.32, 37.10, 12.97, 1.52, 0.37],  # training snr  0
                                [32.10, 31.82, 29.95, 24.17, 12.72, 2.87, 0.40],  # training snr  -6
                                [5.28, 5.24, 5.08, 4.26, 2.84, 1.26, 0.30],  # training snr  -12
                                [0.36, 0.36, 0.39, 0.39, 0.31, 0.35, 0.34]])  # training snr  -18
S2['bz100_imgr_r5'] = np.array([[89.87, 89.07, 85.84, 68.42, 18.96, 2.08, 0.61],  # training snr  18
                                [89.90, 89.08, 85.88, 69.23, 18.24, 1.87, 0.62],  # training snr  12
                                [89.60, 89.02, 86.38, 70.93, 19.04, 1.66, 0.48],  # training snr  6
                                [86.72, 86.13, 83.83, 73.66, 35.52, 4.97, 1.15],  # training snr  0
                                [69.56, 69.21, 66.96, 59.46, 37.70, 10.26, 1.86],  # training snr  -6
                                [19.36, 19.34, 18.64, 16.86, 11.58, 4.84, 1.48],  # training snr  -12
                                [1.49, 1.58, 1.62, 1.61, 1.58, 1.48, 1.33]])  # training snr  -18
S2['bz100_imgr_r10'] =np.array([[95.72, 95.36, 93.63, 81.92, 27.93, 3.61, 1.10],  # training snr  18
                                [95.72, 95.32, 93.54, 81.78, 27.02, 3.64, 1.21],  # training snr  12
                                [95.64, 95.31, 94.01, 83.56, 28.61, 3.26, 0.98],  # training snr  6
                                [94.20, 94.00, 92.75, 86.47, 49.04, 8.31, 2.04],  # training snr  0
                                [83.86, 83.63, 82.33, 76.78, 54.36, 16.13, 3.36],  # training snr  -6
                                [32.54, 32.50, 31.73, 28.45, 19.76, 8.73, 2.76],  # training snr  -12
                                [3.17, 3.06, 3.08, 3.12, 3.14, 2.92, 2.48]])  # training snr  -18
S2['bz100_capr_r1'] =np.array([[75.24, 72.70, 64.10, 38.36, 6.26, 0.60, 0.12],  # training snr  18
                                [75.04, 73.02, 64.80, 39.14, 5.74, 0.38, 0.16],  # training snr  12
                                [74.52, 72.06, 63.66, 39.26, 5.98, 0.34, 0.06],  # training snr  6
                                [66.24, 63.96, 58.04, 41.36, 13.18, 1.52, 0.30],  # training snr  0
                                [40.28, 39.70, 36.76, 27.92, 13.78, 2.90, 0.60],  # training snr  -6
                                [4.88, 4.92, 4.72, 4.32, 2.70, 1.20, 0.30],  # training snr  -12
                                [0.40, 0.38, 0.40, 0.38, 0.42, 0.34, 0.28]])  # training snr  -18
S2['bz100_capr_r5'] =np.array([[93.90, 92.02, 85.60, 61.60, 12.84, 1.16, 0.20],  # training snr  18
                                [94.28, 92.40, 86.18, 62.56, 12.18, 0.88, 0.20],  # training snr  12
                                [93.30, 91.90, 86.12, 62.94, 12.62, 0.70, 0.14],  # training snr  6
                                [90.40, 88.62, 82.78, 65.40, 24.52, 2.80, 0.44],  # training snr  0
                                [72.72, 71.34, 66.34, 52.66, 27.72, 5.78, 1.00],  # training snr  -6
                                [16.00, 16.06, 15.44, 13.52, 8.42, 3.04, 0.88],  # training snr  -12
                                [2.22, 2.22, 2.14, 2.18, 1.58, 1.04, 0.96]])  # training snr  -18
S2['bz100_capr_r10'] =np.array([[97.08, 96.00, 92.30, 73.50, 18.58, 1.74, 0.40],  # training snr  18
                                [97.10, 96.24, 93.04, 73.86, 17.80, 1.48, 0.42],  # training snr  12
                                [96.86, 96.02, 93.18, 74.60, 18.30, 1.20, 0.30],  # training snr  6
                                [95.30, 94.66, 91.66, 78.52, 34.24, 4.30, 0.86],  # training snr  0
                                [84.02, 82.66, 79.18, 67.14, 38.52, 9.24, 1.54],  # training snr  -6
                                [26.26, 26.74, 26.14, 22.48, 13.04, 5.34, 1.46],  # training snr  -12
                                [3.62, 3.76, 3.86, 3.92, 3.20, 2.38, 1.44]])  # training snr  -18
###################################### S3: train without noise, test without noise
S3['bz100_imgr_r1'] = 63.92
S3['bz100_imgr_r5'] = 89.86
S3['bz100_imgr_r10'] = 95.90
S3['bz100_capr_r1'] = 75.40
S3['bz100_capr_r5'] = 93.70
S3['bz100_capr_r10'] = 97.70


###################################### S4: train with noise, test without noise
################## training snr  18     12     6      0     -6    -12    -18
S4['bz100_imgr_r1'] =np.array([64.32, 64.62, 63.68, 56.12, 32.06, 5.34, 0.38])
S4['bz100_imgr_r5'] =np.array([89.80, 90.22, 89.80, 86.82, 69.38, 19.54, 1.50])
S4['bz100_imgr_r10'] =np.array([95.98, 95.84, 95.74, 94.20, 84.00, 32.84, 3.08])
S4['bz100_capr_r1'] =np.array([76.50, 76.10, 75.10, 66.30, 40.00, 4.60, 0.40])
S4['bz100_capr_r5'] =np.array([94.20, 94.80, 94.30, 91.00, 73.50, 16.10, 2.10])
S4['bz100_capr_r10'] =np.array([97.40, 97.80, 97.30, 95.50, 84.90, 26.10, 3.70])

################################################################# plot ###################################################################


#######  S2 compare batch sizes
X, Y = np.meshgrid(S2_SNR, S2_SNR) # X:testing, Y: Training
fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
fig.set_size_inches(16, 9)
for j, ic in enumerate(['img', 'cap']):
     for i, r in enumerate([1, 5, 10]):
         ax[j, i].plot_wireframe(X, Y, S2[f'bz100_{ic}r_r{r}'], color='b')
         ax[j, i].set_xlabel('Test SNR/dB', fontsize=12)
         ax[j, i].set_ylabel('Train SNR/dB', fontsize=12)
         ax[j, i].set_zlabel('Accuracy/%', fontsize=15)
         ax[j, i].set_zlim(0, 100)
         ax[j, i].view_init(20, 300, None)
ax[0, 0].annotate('R@1', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 1].annotate('R@5', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 2].annotate('R@10', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[0, 0].annotate('Image', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
ax[1, 0].annotate('Caption', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
plt.subplots_adjust(wspace=0, hspace=0)
lines_labels = [ax[0, 0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper left', fontsize='xx-large')
plt.show()

#######  S2 find best training snr for given testing snr
# todo: add more data points
# x: testing snr, y: best training snr
fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(16, 9)
for j, ic in enumerate(['img', 'cap']):
     for i, r in enumerate([1, 5, 10]):
         ax[j, i].plot(S2_SNR, S2_SNR[np.argmax(S2[f'bz100_{ic}r_r{r}'], axis=0)], 'b--', linewidth=3.0, label='best train SNR for given test SNR')
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
         ax[j, i].plot(S2_SNR, np.array([S3[f'bz100_{ic}r_r{r}']]*S2_SNR.shape[0]), color='r', linewidth=3.0, label='Scenario A')
         ax[j, i].plot(S2_SNR, S1[f'bz100_{ic}r_r{r}'], 'b--*', linewidth=3.0, markersize=12, label='Scenario B')
         ax[j, i].plot(S2_SNR, S4[f'bz100_{ic}r_r{r}'], 'b--o', linewidth=3.0, markersize=10, label='Scenario C')
         ax[j, i].plot(S2_SNR, np.diag(S2[f'bz100_{ic}r_r{r}']), color='b', linewidth=3.0, label='Scenario D')
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

