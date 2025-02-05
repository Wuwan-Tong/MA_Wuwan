import math

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from optimization.interpolation import upsampl_acc_res_numsym_4D

acc = {}
acc_k = {}
acc_k1 = {}
snr = [18, 12, 6, 0, -6, -12, -18]
################################################################### k layers AE 256 256 #######################################################################################
# dim 512-256 k=1
#                       18   6   -6   -18
acc_k['k1_imgret_r1'] = [67.52, 58.55, 10.72, 0.22]
acc_k['k1_imgret_r5'] = [89.80, 85.33, 31.01, 1.08]
acc_k['k1_imgret_r10'] = [94.34, 91.56, 44.16, 1.91]
acc_k['k1_capret_r1'] = [81.84, 70.40, 11.84, 0.32]
acc_k['k1_capret_r5'] = [95.60, 90.30, 25.34, 0.54]
acc_k['k1_capret_r10'] = [98.16, 95.72, 35.40, 0.98]
# dim 512-384-256 k=2
#                       18   6   -6   -18
acc_k['k2_imgret_r1'] = [61.00, 52.91, 5.99, 0.06]
acc_k['k2_imgret_r5'] = [86.48, 82.41, 21.18, 0.53]
acc_k['k2_imgret_r10'] = [92.47, 89.68, 32.33, 0.98]
acc_k['k2_capret_r1'] = [76.42, 64.38, 6.24, 0.06]
acc_k['k2_capret_r5'] = [94.44, 87.60, 15.34, 0.10]
acc_k['k2_capret_r10'] = [97.26, 93.72, 23.58, 0.32]
# dim 512-427-341-256 k=3
#                       18   6   -6   -18
acc_k['k3_imgret_r1'] = [46.19, 37.90, 3.07, 0.06]
acc_k['k3_imgret_r5'] = [78.48, 71.80, 11.81, 0.44]
acc_k['k3_imgret_r10'] = [87.33, 82.69, 20.06, 0.83]
acc_k['k3_capret_r1'] = [58.76, 46.78, 2.96, 0.02]
acc_k['k3_capret_r5'] = [86.96, 75.14, 8.60, 0.04]
acc_k['k3_capret_r10'] = [92.82, 85.94, 14.10, 0.06]
# dim 512-448-384-320-256 k=4
#                       18   6   -6   -18
acc_k['k4_imgret_r1'] = [26.57, 17.16, 1.66, 0.16]
acc_k['k4_imgret_r5'] = [60.09, 46.24, 6.98, 0.51]
acc_k['k4_imgret_r10'] = [74.44, 62.18, 12.45, 0.96]
acc_k['k4_capret_r1'] = [33.86, 20.48, 1.86, 0.04]
acc_k['k4_capret_r5'] = [66.16, 45.74, 5.04, 0.06]
acc_k['k4_capret_r10'] = [79.22, 59.88, 8.10, 0.22]

# dim 512-461-410-358-307-256 k=5
#                       18   6   -6   -18
acc_k['k5_imgret_r1'] = [15.00, 6.24, 0.68, 0.14]
acc_k['k5_imgret_r5'] = [42.81, 22.00, 3.40, 0.73]
acc_k['k5_imgret_r10'] = [57.74, 34.55, 6.26, 1.60]
acc_k['k5_capret_r1'] = [19.30, 7.10, 0.64, 0.24]
acc_k['k5_capret_r5'] = [45.94, 21.54, 2.20, 0.40]
acc_k['k5_capret_r10'] = [60.42, 32.14, 4.02, 0.68]

################################################################### noise one side #######################################################################################
################################################################ 256 256
acc['symb256_256_imgret_r1_both'] = np.array([68.30] * 7)
acc['symb256_256_imgret_r5_both'] = np.array([89.88] * 7)
acc['symb256_256_imgret_r10_both'] = np.array([94.52] * 7)
acc['symb256_256_capret_r1_both'] = np.array([82.80] * 7)
acc['symb256_256_capret_r5_both'] = np.array([96.50] * 7)
acc['symb256_256_capret_r10_both'] = np.array([98.20] * 7)

acc['symb256_256_imgret_r1_nimg'] = np.array([67.64, 66.48, 62.87, 50.68, 24.64, 6.12, 1.26])
acc['symb256_256_imgret_r5_nimg'] = np.array([89.84, 89.22, 87.76, 80.59, 54.90, 20.49, 4.84])
acc['symb256_256_imgret_r10_nimg'] = np.array([94.55, 94.04, 93.40, 88.63, 68.35, 30.70, 8.44])
acc['symb256_256_capret_r1_nimg'] = np.array([83.28, 82.72, 80.88, 70.82, 40.28, 9.00, 0.90])
acc['symb256_256_capret_r5_nimg'] = np.array([95.94, 95.88, 95.26, 91.20, 69.96, 24.00, 4.34])
acc['symb256_256_capret_r10_nimg'] = np.array([98.10, 98.00, 97.80, 95.40, 80.72, 34.24, 7.42])

acc['symb256_256_imgret_r1_ncap'] = np.array([67.59, 66.78, 64.99, 55.85, 30.92, 7.58, 0.78])
acc['symb256_256_imgret_r5_ncap'] = np.array([89.86, 89.41, 88.13, 83.41, 61.04, 23.63, 4.05])
acc['symb256_256_imgret_r10_ncap'] = np.array([94.49, 94.36, 93.17, 90.12, 73.10, 34.42, 7.43])
acc['symb256_256_capret_r1_ncap'] = np.array([82.40, 81.42, 74.44, 58.64, 26.74, 6.12, 1.36])
acc['symb256_256_capret_r5_ncap'] = np.array([95.62, 94.84, 92.48, 82.46, 47.24, 13.64, 2.64])
acc['symb256_256_capret_r10_ncap'] = np.array([98.08, 97.80, 96.70, 89.96, 60.00, 20.48, 4.26])
################################################################ 256 128
acc['symb256_128_imgret_r1_both'] = np.array([66.68] * 7)
acc['symb256_128_imgret_r5_both'] = np.array([89.24] * 7)
acc['symb256_128_imgret_r10_both'] = np.array([94.22] * 7)
acc['symb256_128_capret_r1_both'] = np.array([82.80] * 7)
acc['symb256_128_capret_r5_both'] = np.array([96.10] * 7)
acc['symb256_128_capret_r10_both'] = np.array([98.00] * 7)

acc['symb256_128_imgret_r1_nimg'] = np.array([66.61, 65.28, 61.44, 49.64, 24.50, 5.88, 1.16])
acc['symb256_128_imgret_r5_nimg'] = np.array([89.34, 88.90, 87.06, 79.83, 55.12, 19.28, 5.08])
acc['symb256_128_imgret_r10_nimg'] = np.array([94.23, 93.90, 92.88, 88.22, 68.85, 29.48, 8.58])
acc['symb256_128_capret_r1_nimg'] = np.array([81.76, 81.66, 79.68, 69.44, 39.96, 7.40, 0.98])
acc['symb256_128_capret_r5_nimg'] = np.array([95.58, 95.38, 94.74, 90.94, 70.38, 21.60, 4.36])
acc['symb256_128_capret_r10_nimg'] = np.array([97.98, 98.18, 97.56, 95.28, 80.76, 31.96, 7.32])

acc['symb256_128_imgret_r1_ncap'] = np.array([66.39, 64.70, 60.68, 45.67, 18.24, 2.78, 0.53])
acc['symb256_128_imgret_r5_ncap'] = np.array([89.25, 88.42, 86.22, 75.72, 44.00, 11.52, 2.86])
acc['symb256_128_imgret_r10_ncap'] = np.array([94.31, 93.70, 92.21, 84.88, 57.18, 18.84, 4.94])
acc['symb256_128_capret_r1_ncap'] = np.array([81.08, 77.62, 68.72, 43.76, 15.34, 2.94, 0.74])
acc['symb256_128_capret_r5_ncap'] = np.array([95.04, 93.68, 88.54, 68.42, 29.40, 6.12, 1.50])
acc['symb256_128_capret_r10_ncap'] = np.array([98.22, 97.26, 93.88, 79.22, 39.78, 9.96, 2.54])
################################################################ 256 64
acc['symb256_64_imgret_r1_both'] = np.array([64.26] * 7)
acc['symb256_64_imgret_r5_both'] = np.array([88.34] * 7)
acc['symb256_64_imgret_r10_both'] = np.array([93.62] * 7)
acc['symb256_64_capret_r1_both'] = np.array([80.10] * 7)
acc['symb256_64_capret_r5_both'] = np.array([94.90] * 7)
acc['symb256_64_capret_r10_both'] = np.array([98.10] * 7)

acc['symb256_64_imgret_r1_nimg'] = np.array([63.82, 62.42, 59.28, 47.40, 23.87, 5.53, 1.08])
acc['symb256_64_imgret_r5_nimg'] = np.array([88.04, 87.58, 85.90, 78.75, 53.24, 18.54, 4.56])
acc['symb256_64_imgret_r10_nimg'] = np.array([93.53, 93.06, 92.08, 87.54, 67.05, 28.48, 8.10])
acc['symb256_64_capret_r1_nimg'] = np.array([80.32, 80.02, 76.62, 67.74, 36.92, 7.72, 0.80])
acc['symb256_64_capret_r5_nimg'] = np.array([95.44, 94.90, 93.64, 89.98, 66.12, 21.84, 4.00])
acc['symb256_64_capret_r10_nimg'] = np.array([97.66, 97.66, 97.40, 95.14, 78.08, 32.18, 7.10])

acc['symb256_64_imgret_r1_ncap'] = np.array([63.62, 61.79, 54.01, 33.69, 9.02, 1.72, 0.43])
acc['symb256_64_imgret_r5_ncap'] = np.array([87.98, 86.83, 82.26, 64.34, 26.56, 6.36, 1.96])
acc['symb256_64_imgret_r10_ncap'] = np.array([93.48, 92.56, 89.46, 75.61, 38.02, 11.08, 3.56])
acc['symb256_64_capret_r1_ncap'] = np.array([76.84, 71.14, 56.82, 29.14, 7.52, 1.66, 0.52])
acc['symb256_64_capret_r5_ncap'] = np.array([93.80, 91.88, 80.98, 50.68, 15.44, 3.70, 1.10])
acc['symb256_64_capret_r10_ncap'] = np.array([97.20, 96.22, 89.10, 63.26, 22.44, 5.52, 1.84])
################################################################ 256 32
acc['symb256_32_imgret_r1_both'] = np.array([55.50] * 7)
acc['symb256_32_imgret_r5_both'] = np.array([84.20] * 7)
acc['symb256_32_imgret_r10_both'] = np.array([91.08] * 7)
acc['symb256_32_capret_r1_both'] = np.array([71.50] * 7)
acc['symb256_32_capret_r5_both'] = np.array([92.40] * 7)
acc['symb256_32_capret_r10_both'] = np.array([95.20] * 7)

acc['symb256_32_imgret_r1_nimg'] = np.array([55.80, 53.97, 50.58, 41.28, 20.88, 5.77, 1.01])
acc['symb256_32_imgret_r5_nimg'] = np.array([84.24, 83.22, 80.98, 74.17, 49.85, 18.39, 4.42])
acc['symb256_32_imgret_r10_nimg'] = np.array([91.03, 90.26, 89.08, 84.30, 64.24, 28.36, 7.56])
acc['symb256_32_capret_r1_nimg'] = np.array([70.26, 69.34, 67.58, 59.26, 34.98, 7.32, 0.78])
acc['symb256_32_capret_r5_nimg'] = np.array([92.94, 91.40, 90.72, 85.90, 64.66, 22.40, 3.56])
acc['symb256_32_capret_r10_nimg'] = np.array([96.60, 95.48, 95.42, 92.78, 76.88, 32.40, 6.42])

acc['symb256_32_imgret_r1_ncap'] = np.array([54.84, 51.58, 41.09, 18.05, 3.71, 0.78, 0.20])
acc['symb256_32_imgret_r5_ncap'] = np.array([83.25, 80.61, 73.08, 44.55, 13.57, 3.40, 1.19])
acc['symb256_32_imgret_r10_ncap'] = np.array([90.25, 88.06, 82.59, 58.14, 21.40, 6.25, 2.35])
acc['symb256_32_capret_r1_ncap'] = np.array([65.50, 57.44, 39.22, 13.82, 3.28, 0.70, 0.30])
acc['symb256_32_capret_r5_ncap'] = np.array([89.04, 81.96, 64.52, 27.94, 7.18, 1.58, 0.60])
acc['symb256_32_capret_r10_ncap'] = np.array([93.96, 90.04, 76.60, 39.36, 11.00, 2.76, 1.02])
################################################################ 256 16
acc['symb256_16_imgret_r1_both'] = np.array([39.20] * 7)
acc['symb256_16_imgret_r5_both'] = np.array([72.92] * 7)
acc['symb256_16_imgret_r10_both'] = np.array([83.70] * 7)
acc['symb256_16_capret_r1_both'] = np.array([48.60] * 7)
acc['symb256_16_capret_r5_both'] = np.array([80.70] * 7)
acc['symb256_16_capret_r10_both'] = np.array([89.70] * 7)

acc['symb256_16_imgret_r1_nimg'] = np.array([38.46, 37.85, 33.33, 28.54, 16.64, 4.48, 0.84])
acc['symb256_16_imgret_r5_nimg'] = np.array([72.05, 71.86, 67.92, 61.32, 43.16, 15.90, 3.78])
acc['symb256_16_imgret_r10_nimg'] = np.array([82.75, 82.50, 80.02, 74.59, 57.76, 25.48, 6.81])
acc['symb256_16_capret_r1_nimg'] = np.array([46.68, 49.44, 46.48, 40.14, 26.32, 5.94, 0.64])
acc['symb256_16_capret_r5_nimg'] = np.array([78.90, 80.70, 76.88, 72.46, 54.70, 18.66, 3.34])
acc['symb256_16_capret_r10_nimg'] = np.array([88.12, 90.04, 86.76, 83.94, 68.58, 27.50, 5.84])

acc['symb256_16_imgret_r1_ncap'] = np.array([37.85, 34.62, 23.41, 8.51, 1.80, 0.45, 0.22])
acc['symb256_16_imgret_r5_ncap'] = np.array([71.09, 68.03, 54.31, 25.47, 7.10, 1.94, 0.93])
acc['symb256_16_imgret_r10_ncap'] = np.array([81.97, 79.69, 67.92, 37.09, 11.49, 3.75, 2.01])
acc['symb256_16_capret_r1_ncap'] = np.array([42.42, 34.48, 18.88, 6.68, 1.78, 0.38, 0.22])
acc['symb256_16_capret_r5_ncap'] = np.array([3.18, 62.24, 38.60, 13.72, 3.70, 0.94, 0.38])
acc['symb256_16_capret_r10_ncap'] = np.array([84.12, 75.56, 51.34, 20.72, 5.62, 1.80, 0.62])
################################################################ 128 256
acc['symb128_256_imgret_r1_both'] = np.array([66.86] * 7)
acc['symb128_256_imgret_r5_both'] = np.array([89.22] * 7)
acc['symb128_256_imgret_r10_both'] = np.array([94.28] * 7)
acc['symb128_256_capret_r1_both'] = np.array([82.30] * 7)
acc['symb128_256_capret_r5_both'] = np.array([95.40] * 7)
acc['symb128_256_capret_r10_both'] = np.array([98.00] * 7)

acc['symb128_256_imgret_r1_nimg'] = np.array([65.85, 63.90, 57.00, 39.65, 14.30, 3.06, 0.72])
acc['symb128_256_imgret_r5_nimg'] = np.array([89.42, 88.35, 85.12, 71.92, 37.94, 11.10, 3.04])
acc['symb128_256_imgret_r10_nimg'] = np.array([94.34, 93.59, 91.52, 82.68, 51.70, 17.84, 5.44])
acc['symb128_256_capret_r1_nimg'] = np.array([81.48, 80.68, 76.12, 59.74, 22.86, 3.24, 0.52])
acc['symb128_256_capret_r5_nimg'] = np.array([96.10, 95.48, 94.04, 85.66, 48.30, 11.82, 2.30])
acc['symb128_256_capret_r10_nimg'] = np.array([98.62, 98.06, 97.46, 91.84, 61.10, 18.38, 4.30])

acc['symb128_256_imgret_r1_ncap'] = np.array([65.62, 65.92, 63.61, 53.86, 30.77, 7.06, 0.75])
acc['symb128_256_imgret_r5_ncap'] = np.array([88.82, 89.05, 87.80, 82.14, 61.02, 22.38, 4.15])
acc['symb128_256_imgret_r10_ncap'] = np.array([94.01, 94.09, 93.01, 89.38, 73.24, 33.08, 7.79])
acc['symb128_256_capret_r1_ncap'] = np.array([81.60, 79.76, 73.14, 56.16, 26.92, 6.42, 1.26])
acc['symb128_256_capret_r5_ncap'] = np.array([95.36, 94.76, 91.48, 80.02, 47.40, 13.32, 2.54])
acc['symb128_256_capret_r10_ncap'] = np.array([97.76, 97.82, 96.78, 88.48, 60.78, 19.08, 4.08])
################################################################ 128 128
acc['symb128_128_imgret_r1_both'] = np.array([65.36] * 7)
acc['symb128_128_imgret_r5_both'] = np.array([88.68] * 7)
acc['symb128_128_imgret_r10_both'] = np.array([93.64] * 7)
acc['symb128_128_capret_r1_both'] = np.array([81.40] * 7)
acc['symb128_128_capret_r5_both'] = np.array([95.90] * 7)
acc['symb128_128_capret_r10_both'] = np.array([98.20] * 7)

acc['symb128_128_imgret_r1_nimg'] = np.array([64.55, 62.92, 56.50, 38.10, 13.06, 2.88, 0.61])
acc['symb128_128_imgret_r5_nimg'] = np.array([88.38, 87.69, 84.52, 70.97, 35.90, 11.05, 2.93])
acc['symb128_128_imgret_r10_nimg'] = np.array([93.73, 93.15, 91.32, 81.86, 49.52, 17.96, 5.36])
acc['symb128_128_capret_r1_nimg'] = np.array([79.22, 79.72, 75.54, 57.92, 21.22, 3.58, 0.50])
acc['symb128_128_capret_r5_nimg'] = np.array([95.38, 94.54, 93.52, 84.52, 46.40, 11.48, 2.10])
acc['symb128_128_capret_r10_nimg'] = np.array([97.88, 97.72, 96.92, 91.00, 59.20, 17.82, 3.90])

acc['symb128_128_imgret_r1_ncap'] = np.array([65.56, 63.89, 59.54, 45.19, 18.13, 3.02, 0.59])
acc['symb128_128_imgret_r5_ncap'] = np.array([88.40, 88.08, 85.83, 75.48, 43.67, 12.04, 2.35])
acc['symb128_128_imgret_r10_ncap'] = np.array([93.76, 93.49, 91.90, 84.87, 56.69, 19.44, 4.73])
acc['symb128_128_capret_r1_ncap'] = np.array([79.28, 76.68, 66.18, 43.20, 13.92, 2.98, 0.70])
acc['symb128_128_capret_r5_ncap'] = np.array([95.08, 93.88, 87.82, 67.20, 28.52, 6.62, 1.36])
acc['symb128_128_capret_r10_ncap'] = np.array([98.10, 97.42, 93.96, 78.60, 39.26, 10.32, 2.08])
################################################################ 128 64
acc['symb128_64_imgret_r1_both'] = np.array([63.72] * 7)
acc['symb128_64_imgret_r5_both'] = np.array([87.22] * 7)
acc['symb128_64_imgret_r10_both'] = np.array([92.86] * 7)
acc['symb128_64_capret_r1_both'] = np.array([77.30] * 7)
acc['symb128_64_capret_r5_both'] = np.array([94.50] * 7)
acc['symb128_64_capret_r10_both'] = np.array([97.70] * 7)

acc['symb128_64_imgret_r1_nimg'] = np.array([61.43, 60.12, 53.90, 36.46, 12.96, 2.76, 0.69])
acc['symb128_64_imgret_r5_nimg'] = np.array([86.62, 86.13, 82.59, 69.12, 35.72, 10.46, 2.67])
acc['symb128_64_imgret_r10_nimg'] = np.array([92.66, 92.19, 90.10, 80.22, 49.31, 17.12, 5.08])
acc['symb128_64_capret_r1_nimg'] = np.array([78.94, 76.72, 72.18, 54.18, 21.40, 3.68, 0.42])
acc['symb128_64_capret_r5_nimg'] = np.array([93.98, 94.08, 92.08, 83.42, 46.06, 10.80, 2.36])
acc['symb128_64_capret_r10_nimg'] = np.array([96.96, 97.24, 96.54, 91.18, 58.70, 17.74, 4.08])

acc['symb128_64_imgret_r1_ncap'] = np.array([62.39, 59.94, 53.00, 32.06, 8.46, 1.54, 0.31])
acc['symb128_64_imgret_r5_ncap'] = np.array([87.07, 85.82, 81.99, 62.54, 25.54, 6.19, 1.56])
acc['symb128_64_imgret_r10_ncap'] = np.array([92.69, 91.67, 89.13, 74.64, 37.23, 10.72, 3.08])
acc['symb128_64_capret_r1_ncap'] = np.array([76.38, 71.14, 56.34, 28.08, 6.94, 1.48, 0.42])
acc['symb128_64_capret_r5_ncap'] = np.array([94.04, 90.94, 79.90, 48.66, 14.30, 3.28, 0.78])
acc['symb128_64_capret_r10_ncap'] = np.array([97.10, 95.78, 88.68, 62.08, 21.76, 5.34, 1.56])
################################################################ 128 32
acc['symb128_32_imgret_r1_both'] = np.array([54.98] * 7)
acc['symb128_32_imgret_r5_both'] = np.array([83.64] * 7)
acc['symb128_32_imgret_r10_both'] = np.array([90.96] * 7)
acc['symb128_32_capret_r1_both'] = np.array([69.00] * 7)
acc['symb128_32_capret_r5_both'] = np.array([91.20] * 7)
acc['symb128_32_capret_r10_both'] = np.array([94.80] * 7)

acc['symb128_32_imgret_r1_nimg'] = np.array([54.85, 53.50, 46.64, 31.77, 12.35, 2.72, 0.68])
acc['symb128_32_imgret_r5_nimg'] = np.array([83.32, 82.70, 78.61, 65.18, 34.15, 10.49, 2.96])
acc['symb128_32_imgret_r10_nimg'] = np.array([90.81, 89.93, 87.40, 77.58, 47.63, 17.42, 5.25])
acc['symb128_32_capret_r1_nimg'] = np.array([71.60, 69.80, 64.82, 50.16, 19.40, 3.36, 0.56])
acc['symb128_32_capret_r5_nimg'] = np.array([91.64, 91.88, 88.60, 80.00, 43.72, 11.06, 2.32])
acc['symb128_32_capret_r10_nimg'] = np.array([96.14, 95.78, 94.26, 88.32, 57.54, 17.74, 4.26])

acc['symb128_32_imgret_r1_ncap'] = np.array([54.27, 51.24, 40.25, 17.53, 3.48, 0.70, 0.22])
acc['symb128_32_imgret_r5_ncap'] = np.array([82.93, 81.34, 72.19, 43.27, 12.68, 3.18, 1.34])
acc['symb128_32_imgret_r10_ncap'] = np.array([89.98, 89.16, 82.55, 57.14, 20.82, 5.77, 2.72])
acc['symb128_32_capret_r1_ncap'] = np.array([66.52, 57.94, 37.58, 14.30, 3.42, 0.74, 0.38])
acc['symb128_32_capret_r5_ncap'] = np.array([89.22, 83.28, 63.14, 27.48, 7.00, 1.60, 0.74])
acc['symb128_32_capret_r10_ncap'] = np.array([94.74, 90.90, 75.00, 38.04, 10.62, 2.70, 1.24])
################################################################ 128 16
acc['symb128_16_imgret_r1_both'] = np.array([39.48] * 7)
acc['symb128_16_imgret_r5_both'] = np.array([72.52] * 7)
acc['symb128_16_imgret_r10_both'] = np.array([83.10] * 7)
acc['symb128_16_capret_r1_both'] = np.array([47.40] * 7)
acc['symb128_16_capret_r5_both'] = np.array([79.60] * 7)
acc['symb128_16_capret_r10_both'] = np.array([88.20] * 7)

acc['symb128_16_imgret_r1_nimg'] = np.array([39.05, 36.80, 31.81, 22.58, 10.19, 2.56, 0.64])
acc['symb128_16_imgret_r5_nimg'] = np.array([72.51, 71.03, 65.09, 54.27, 30.75, 9.16, 2.82])
acc['symb128_16_imgret_r10_nimg'] = np.array([83.60, 81.85, 78.01, 68.88, 43.96, 15.75, 5.31])
acc['symb128_16_capret_r1_nimg'] = np.array([50.14, 46.32, 43.12, 36.42, 16.40, 2.76, 0.54])
acc['symb128_16_capret_r5_nimg'] = np.array([79.38, 78.68, 76.60, 67.64, 39.22, 9.00, 1.96])
acc['symb128_16_capret_r10_nimg'] = np.array([88.14, 88.16, 86.60, 78.92, 52.80, 15.36, 3.48])

acc['symb128_16_imgret_r1_ncap'] = np.array([36.57, 33.53, 23.54, 8.85, 1.60, 0.40, 0.21])
acc['symb128_16_imgret_r5_ncap'] = np.array([70.56, 66.37, 54.57, 25.41, 6.62, 1.98, 0.98])
acc['symb128_16_imgret_r10_ncap'] = np.array([81.64, 78.39, 68.21, 37.05, 11.32, 3.80, 1.78])
acc['symb128_16_capret_r1_ncap'] = np.array([41.74, 32.46, 19.20, 6.46, 1.68, 0.50, 0.26])
acc['symb128_16_capret_r5_ncap'] = np.array([72.22, 59.20, 38.90, 14.62, 3.74, 0.96, 0.38])
acc['symb128_16_capret_r10_ncap'] = np.array([82.98, 72.80, 51.94, 21.20, 6.14, 1.64, 0.80])
################################################################ 64 256
acc['symb64_256_imgret_r1_both'] = np.array([64.26] * 7)
acc['symb64_256_imgret_r5_both'] = np.array([87.88] * 7)
acc['symb64_256_imgret_r10_both'] = np.array([93.46] * 7)
acc['symb64_256_capret_r1_both'] = np.array([81.00] * 7)
acc['symb64_256_capret_r5_both'] = np.array([95.20] * 7)
acc['symb64_256_capret_r10_both'] = np.array([97.70] * 7)

acc['symb64_256_imgret_r1_nimg'] = np.array([62.96, 59.22, 48.84, 25.46, 7.01, 1.26, 0.46])
acc['symb64_256_imgret_r5_nimg'] = np.array([87.55, 86.18, 79.53, 55.75, 22.30, 5.90, 2.00])
acc['symb64_256_imgret_r10_nimg'] = np.array([93.24, 92.46, 87.83, 69.60, 32.78, 10.10, 3.84])
acc['symb64_256_capret_r1_nimg'] = np.array([80.42, 77.80, 68.42, 40.68, 10.46, 1.48, 0.32])
acc['symb64_256_capret_r5_nimg'] = np.array([94.78, 94.72, 90.20, 70.50, 27.36, 5.16, 1.58])
acc['symb64_256_capret_r10_nimg'] = np.array([97.48, 97.54, 95.48, 81.14, 38.46, 8.80, 2.92])

acc['symb64_256_imgret_r1_ncap'] = np.array([64.31, 63.71, 60.80, 52.46, 29.20, 6.28, 0.96])
acc['symb64_256_imgret_r5_ncap'] = np.array([88.10, 87.82, 86.22, 81.62, 59.38, 20.49, 3.96])
acc['symb64_256_imgret_r10_ncap'] = np.array([93.54, 93.41, 92.10, 89.11, 71.79, 30.46, 7.24])
acc['symb64_256_capret_r1_ncap'] = np.array([78.72, 77.36, 70.82, 54.66, 24.98, 6.18, 0.96])
acc['symb64_256_capret_r5_ncap'] = np.array([95.00, 94.44, 91.04, 79.58, 45.36, 11.92, 2.12])
acc['symb64_256_capret_r10_ncap'] = np.array([97.90, 97.32, 95.44, 88.56, 57.96, 17.24, 3.52])
################################################################ 64 128
acc['symb64_128_imgret_r1_both'] = np.array([63.48] * 7)
acc['symb64_128_imgret_r5_both'] = np.array([87.52] * 7)
acc['symb64_128_imgret_r10_both'] = np.array([93.46] * 7)
acc['symb64_128_capret_r1_both'] = np.array([79.40] * 7)
acc['symb64_128_capret_r5_both'] = np.array([95.80] * 7)
acc['symb64_128_capret_r10_both'] = np.array([98.00] * 7)

acc['symb64_128_imgret_r1_nimg'] = np.array([61.52, 58.34, 48.00, 24.54, 6.72, 1.42, 0.50])
acc['symb64_128_imgret_r5_nimg'] = np.array([86.91, 85.41, 78.76, 54.52, 21.68, 6.10, 2.02])
acc['symb64_128_imgret_r10_nimg'] = np.array([92.84, 91.86, 87.30, 68.36, 32.48, 10.60, 3.70])
acc['symb64_128_capret_r1_nimg'] = np.array([77.74, 76.20, 67.50, 39.84, 10.76, 1.62, 0.24])
acc['symb64_128_capret_r5_nimg'] = np.array([94.10, 93.96, 90.54, 69.86, 27.28, 5.56, 1.48])
acc['symb64_128_capret_r10_nimg'] = np.array([96.96, 97.08, 95.46, 80.20, 37.84, 9.28, 2.72])

acc['symb64_128_imgret_r1_ncap'] = np.array([62.93, 61.55, 57.21, 42.74, 17.07, 2.70, 0.61])
acc['symb64_128_imgret_r5_ncap'] = np.array([87.30, 86.62, 84.58, 73.91, 42.39, 11.06, 2.87])
acc['symb64_128_imgret_r10_ncap'] = np.array([93.29, 92.59, 91.19, 83.53, 55.98, 18.45, 4.93])
acc['symb64_128_capret_r1_ncap'] = np.array([77.30, 74.38, 63.28, 40.06, 13.94, 2.94, 0.88])
acc['symb64_128_capret_r5_ncap'] = np.array([94.50, 92.42, 86.72, 65.40, 27.36, 5.86, 1.40])
acc['symb64_128_capret_r10_ncap'] = np.array([97.52, 96.46, 93.26, 76.50, 37.90, 9.64, 2.58])
################################################################ 64 64
acc['symb64_64_imgret_r1_both'] = np.array([60.02] * 7)
acc['symb64_64_imgret_r5_both'] = np.array([85.98] * 7)
acc['symb64_64_imgret_r10_both'] = np.array([92.02] * 7)
acc['symb64_64_capret_r1_both'] = np.array([73.80] * 7)
acc['symb64_64_capret_r5_both'] = np.array([93.60] * 7)
acc['symb64_64_capret_r10_both'] = np.array([97.70] * 7)

acc['symb64_64_imgret_r1_nimg'] = np.array([59.02, 55.40, 45.10, 22.83, 6.53, 1.33, 0.36])
acc['symb64_64_imgret_r5_nimg'] = np.array([85.79, 83.88, 76.95, 53.08, 21.11, 5.65, 1.97])
acc['symb64_64_imgret_r10_nimg'] = np.array([92.01, 90.88, 86.38, 67.15, 32.02, 9.79, 3.78])
acc['symb64_64_capret_r1_nimg'] = np.array([76.14, 73.30, 65.04, 37.96, 9.68, 1.34, 0.42])
acc['symb64_64_capret_r5_nimg'] = np.array([93.92, 93.26, 89.56, 67.76, 26.08, 5.20, 1.46])
acc['symb64_64_capret_r10_nimg'] = np.array([97.00, 96.26, 94.38, 79.60, 36.56, 8.72, 2.86])

acc['symb64_64_imgret_r1_ncap'] = np.array([59.77, 57.54, 51.20, 29.79, 8.44, 1.37, 0.30])
acc['symb64_64_imgret_r5_ncap'] = np.array([86.20, 84.75, 80.56, 60.71, 26.01, 6.39, 1.48])
acc['symb64_64_imgret_r10_ncap'] = np.array([91.97, 91.40, 88.22, 72.95, 37.43, 10.81, 3.01])
acc['symb64_64_capret_r1_ncap'] = np.array([74.38, 69.00, 53.64, 25.68, 7.22, 1.44, 0.46])
acc['symb64_64_capret_r5_ncap'] = np.array([92.60, 89.66, 78.36, 46.16, 15.00, 3.26, 0.88])
acc['symb64_64_capret_r10_ncap'] = np.array([96.64, 94.62, 87.58, 60.04, 21.24, 5.40, 1.52])
################################################################ 64 32
acc['symb64_32_imgret_r1_both'] = np.array([53.94] * 7)
acc['symb64_32_imgret_r5_both'] = np.array([82.78] * 7)
acc['symb64_32_imgret_r10_both'] = np.array([90.36] * 7)
acc['symb64_32_capret_r1_both'] = np.array([70.10] * 7)
acc['symb64_32_capret_r5_both'] = np.array([92.20] * 7)
acc['symb64_32_capret_r10_both'] = np.array([96.10] * 7)

acc['symb64_32_imgret_r1_nimg'] = np.array([53.10, 50.08, 41.39, 22.85, 6.75, 1.32, 0.42])
acc['symb64_32_imgret_r5_nimg'] = np.array([82.87, 80.52, 74.04, 52.09, 21.06, 5.40, 1.90])
acc['symb64_32_imgret_r10_nimg'] = np.array([89.92, 88.43, 84.28, 66.09, 31.49, 9.72, 3.59])
acc['symb64_32_capret_r1_nimg'] = np.array([68.38, 67.06, 59.06, 36.96, 9.10, 1.44, 0.28])
acc['symb64_32_capret_r5_nimg'] = np.array([91.48, 89.60, 85.70, 66.42, 25.52, 5.48, 1.62])
acc['symb64_32_capret_r10_nimg'] = np.array([95.30, 94.58, 92.54, 78.10, 35.32, 8.84, 2.90])

acc['symb64_32_imgret_r1_ncap'] = np.array([52.86, 51.62, 39.60, 16.71, 3.71, 0.68, 0.40])
acc['symb64_32_imgret_r5_ncap'] = np.array([82.28, 81.13, 70.63, 42.32, 13.74, 3.42, 1.19])
acc['symb64_32_imgret_r10_ncap'] = np.array([89.66, 88.84, 81.17, 56.43, 21.88, 6.27, 2.69])
acc['symb64_32_capret_r1_ncap'] = np.array([65.14, 56.74, 36.30, 12.80, 3.28, 0.74, 0.30])
acc['symb64_32_capret_r5_ncap'] = np.array([88.04, 82.48, 62.58, 26.56, 6.98, 1.72, 0.56])
acc['symb64_32_capret_r10_ncap'] = np.array([93.70, 90.40, 74.60, 37.60, 10.82, 2.88, 1.06])
################################################################ 64 16
acc['symb64_16_imgret_r1_both'] = np.array([37.84] * 7)
acc['symb64_16_imgret_r5_both'] = np.array([72.08] * 7)
acc['symb64_16_imgret_r10_both'] = np.array([82.90] * 7)
acc['symb64_16_capret_r1_both'] = np.array([48.70] * 7)
acc['symb64_16_capret_r5_both'] = np.array([78.30] * 7)
acc['symb64_16_capret_r10_both'] = np.array([88.10] * 7)

acc['symb64_16_imgret_r1_nimg'] = np.array([36.82, 34.29, 27.99, 16.55, 6.08, 1.37, 0.43])
acc['symb64_16_imgret_r5_nimg'] = np.array([71.05, 67.81, 61.74, 43.93, 19.66, 5.25, 1.75])
acc['symb64_16_imgret_r10_nimg'] = np.array([82.41, 79.77, 75.42, 58.58, 30.16, 9.25, 3.27])
acc['symb64_16_capret_r1_nimg'] = np.array([47.06, 45.42, 41.86, 27.02, 9.10, 1.04, 0.36])
acc['symb64_16_capret_r5_nimg'] = np.array([79.42, 77.34, 72.98, 56.38, 23.78, 4.72, 1.62])
acc['symb64_16_capret_r10_nimg'] = np.array([88.74, 86.54, 84.34, 70.40, 34.22, 8.52, 3.00])

acc['symb64_16_imgret_r1_ncap'] = np.array([37.09, 32.58, 23.01, 8.28, 1.49, 0.31, 0.17])
acc['symb64_16_imgret_r5_ncap'] = np.array([70.92, 66.72, 53.42, 25.18, 6.60, 2.10, 0.84])
acc['symb64_16_imgret_r10_ncap'] = np.array([81.80, 78.42, 67.49, 37.01, 11.58, 3.94, 1.84])
acc['symb64_16_capret_r1_ncap'] = np.array([42.38, 32.24, 18.70, 6.58, 1.40, 0.48, 0.32])
acc['symb64_16_capret_r5_ncap'] = np.array([72.30, 60.18, 39.24, 14.20, 3.36, 0.90, 0.66])
acc['symb64_16_capret_r10_ncap'] = np.array([83.72, 73.48, 51.82, 20.44, 5.60, 1.62, 0.92])
################################################################ 32 256
acc['symb32_256_imgret_r1_both'] = np.array([56.40] * 7)
acc['symb32_256_imgret_r5_both'] = np.array([84.52] * 7)
acc['symb32_256_imgret_r10_both'] = np.array([90.96] * 7)
acc['symb32_256_capret_r1_both'] = np.array([70.60] * 7)
acc['symb32_256_capret_r5_both'] = np.array([92.90] * 7)
acc['symb32_256_capret_r10_both'] = np.array([96.90] * 7)

acc['symb32_256_imgret_r1_nimg'] = np.array([54.30, 48.60, 33.85, 14.50, 3.34, 0.74, 0.43])
acc['symb32_256_imgret_r5_nimg'] = np.array([83.34, 79.41, 66.96, 37.50, 11.32, 3.33, 1.48])
acc['symb32_256_imgret_r10_nimg'] = np.array([90.08, 88.04, 78.81, 51.52, 18.54, 6.37, 2.74])
acc['symb32_256_capret_r1_nimg'] = np.array([70.40, 66.94, 53.28, 23.20, 3.68, 0.74, 0.18])
acc['symb32_256_capret_r5_nimg'] = np.array([91.20, 90.28, 81.50, 48.42, 12.36, 2.58, 1.00])
acc['symb32_256_capret_r10_nimg'] = np.array([96.16, 94.80, 89.18, 62.06, 18.40, 4.80, 2.02])

acc['symb32_256_imgret_r1_ncap'] = np.array([55.42, 55.60, 53.90, 46.12, 26.96, 6.23, 0.94])
acc['symb32_256_imgret_r5_ncap'] = np.array([83.92, 83.33, 82.53, 77.04, 58.16, 20.74, 4.04])
acc['symb32_256_imgret_r10_ncap'] = np.array([90.51, 90.43, 89.72, 86.05, 70.69, 31.56, 7.34])
acc['symb32_256_capret_r1_ncap'] = np.array([70.30, 68.28, 62.70, 48.24, 22.50, 5.52, 0.90])
acc['symb32_256_capret_r5_ncap'] = np.array([92.72, 90.42, 87.12, 74.42, 43.64, 11.78, 2.02])
acc['symb32_256_capret_r10_ncap'] = np.array([96.40, 95.22, 93.50, 84.64, 56.18, 17.86, 3.52])
################################################################ 32 128
acc['symb32_128_imgret_r1_both'] = np.array([55.18] * 7)
acc['symb32_128_imgret_r5_both'] = np.array([83.36] * 7)
acc['symb32_128_imgret_r10_both'] = np.array([90.98] * 7)
acc['symb32_128_capret_r1_both'] = np.array([69.70] * 7)
acc['symb32_128_capret_r5_both'] = np.array([91.40] * 7)
acc['symb32_128_capret_r10_both'] = np.array([95.50] * 7)

acc['symb32_128_imgret_r1_nimg'] = np.array([53.25, 47.64, 33.37, 13.70, 3.52, 0.81, 0.24])
acc['symb32_128_imgret_r5_nimg'] = np.array([82.62, 79.55, 67.18, 37.18, 11.70, 3.55, 1.26])
acc['symb32_128_imgret_r10_nimg'] = np.array([90.01, 88.34, 79.40, 50.94, 18.98, 6.37, 2.59])
acc['symb32_128_capret_r1_nimg'] = np.array([68.84, 66.00, 53.90, 23.88, 4.42, 0.60, 0.20])
acc['symb32_128_capret_r5_nimg'] = np.array([92.50, 89.92, 81.92, 48.98, 12.82, 2.68, 1.00])
acc['symb32_128_capret_r10_nimg'] = np.array([96.24, 94.90, 89.98, 61.84, 19.46, 5.14, 1.82])

acc['symb32_128_imgret_r1_ncap'] = np.array([55.21, 53.87, 51.11, 38.67, 15.51, 3.21, 0.59])
acc['symb32_128_imgret_r5_ncap'] = np.array([84.05, 83.27, 81.04, 70.63, 40.31, 11.38, 2.40])
acc['symb32_128_imgret_r10_ncap'] = np.array([90.79, 90.36, 88.73, 80.68, 54.27, 18.01, 4.72])
acc['symb32_128_capret_r1_ncap'] = np.array([70.08, 65.68, 57.64, 36.12, 12.12, 2.92, 0.62])
acc['symb32_128_capret_r5_ncap'] = np.array([91.50, 88.24, 82.82, 60.58, 25.46, 6.06, 1.24])
acc['symb32_128_capret_r10_ncap'] = np.array([95.58, 94.26, 90.56, 73.14, 35.72, 9.96, 2.24])
################################################################ 32 64
acc['symb32_64_imgret_r1_both'] = np.array([55.86] * 7)
acc['symb32_64_imgret_r5_both'] = np.array([83.48] * 7)
acc['symb32_64_imgret_r10_both'] = np.array([90.50] * 7)
acc['symb32_64_capret_r1_both'] = np.array([69.50] * 7)
acc['symb32_64_capret_r5_both'] = np.array([91.20] * 7)
acc['symb32_64_capret_r10_both'] = np.array([96.00] * 7)

acc['symb32_64_imgret_r1_nimg'] = np.array([52.98, 47.24, 32.82, 13.20, 3.09, 0.85, 0.30])
acc['symb32_64_imgret_r5_nimg'] = np.array([82.42, 78.71, 65.86, 35.07, 11.10, 3.50, 1.32])
acc['symb32_64_imgret_r10_nimg'] = np.array([89.85, 87.43, 78.29, 48.87, 18.12, 6.35, 2.58])
acc['symb32_64_capret_r1_nimg'] = np.array([68.30, 65.42, 50.34, 21.60, 3.92, 0.68, 0.28])
acc['symb32_64_capret_r5_nimg'] = np.array([91.42, 88.90, 79.46, 46.00, 12.18, 2.76, 0.90])
acc['symb32_64_capret_r10_nimg'] = np.array([95.48, 93.92, 88.08, 59.40, 18.78, 5.00, 1.92])

acc['symb32_64_imgret_r1_ncap'] = np.array([53.59, 52.02, 45.36, 27.97, 8.27, 1.62, 0.35])
acc['symb32_64_imgret_r5_ncap'] = np.array([82.85, 81.68, 76.52, 59.17, 25.36, 6.16, 1.62])
acc['symb32_64_imgret_r10_ncap'] = np.array([90.34, 89.30, 85.59, 71.56, 36.91, 10.53, 3.16])
acc['symb32_64_capret_r1_ncap'] = np.array([66.76, 61.96, 47.70, 23.30, 6.70, 1.38, 0.52])
acc['symb32_64_capret_r5_ncap'] = np.array([88.74, 86.36, 74.10, 45.02, 14.48, 3.40, 1.06])
acc['symb32_64_capret_r10_ncap'] = np.array([94.32, 93.20, 84.42, 57.38, 21.30, 5.42, 1.66])
################################################################ 32 32
acc['symb32_32_imgret_r1_both'] = np.array([51.84] * 7)
acc['symb32_32_imgret_r5_both'] = np.array([81.12] * 7)
acc['symb32_32_imgret_r10_both'] = np.array([89.06] * 7)
acc['symb32_32_capret_r1_both'] = np.array([64.60] * 7)
acc['symb32_32_capret_r5_both'] = np.array([88.90] * 7)
acc['symb32_32_capret_r10_both'] = np.array([94.60] * 7)

acc['symb32_32_imgret_r1_nimg'] = np.array([47.89, 43.92, 29.06, 12.46, 2.98, 0.73, 0.28])
acc['symb32_32_imgret_r5_nimg'] = np.array([79.57, 76.55, 61.64, 34.69, 10.89, 3.12, 1.41])
acc['symb32_32_imgret_r10_nimg'] = np.array([87.73, 85.86, 74.89, 48.79, 17.81, 6.01, 2.72])
acc['symb32_32_capret_r1_nimg'] = np.array([64.30, 62.14, 46.16, 20.30, 3.62, 0.50, 0.20])
acc['symb32_32_capret_r5_nimg'] = np.array([89.00, 87.00, 76.72, 45.98, 11.20, 3.02, 1.14])
acc['symb32_32_capret_r10_nimg'] = np.array([94.82, 92.28, 85.94, 59.46, 17.48, 5.08, 2.04])

acc['symb32_32_imgret_r1_ncap'] = np.array([49.47, 48.17, 36.93, 15.84, 3.50, 0.65, 0.30])
acc['symb32_32_imgret_r5_ncap'] = np.array([79.93, 79.42, 68.62, 40.64, 12.70, 3.12, 1.26])
acc['symb32_32_imgret_r10_ncap'] = np.array([88.04, 87.64, 79.91, 54.40, 20.64, 5.68, 2.59])
acc['symb32_32_capret_r1_ncap'] = np.array([60.76, 53.24, 33.38, 12.98, 3.38, 0.86, 0.28])
acc['symb32_32_capret_r5_ncap'] = np.array([85.68, 80.94, 59.26, 26.00, 6.96, 1.58, 0.70])
acc['symb32_32_capret_r10_ncap'] = np.array([92.00, 89.12, 72.20, 35.94, 10.80, 2.52, 0.98])
################################################################ 32 16
acc['symb32_16_imgret_r1_both'] = np.array([37.64] * 7)
acc['symb32_16_imgret_r5_both'] = np.array([71.84] * 7)
acc['symb32_16_imgret_r10_both'] = np.array([83.22] * 7)
acc['symb32_16_capret_r1_both'] = np.array([49.00] * 7)
acc['symb32_16_capret_r5_both'] = np.array([79.20] * 7)
acc['symb32_16_capret_r10_both'] = np.array([88.10] * 7)

acc['symb32_16_imgret_r1_nimg'] = np.array([36.12, 30.89, 23.66, 10.26, 2.80, 0.76, 0.35])
acc['symb32_16_imgret_r5_nimg'] = np.array([69.86, 65.40, 55.32, 30.36, 9.98, 3.29, 1.38])
acc['symb32_16_imgret_r10_nimg'] = np.array([81.28, 78.12, 69.77, 43.30, 16.44, 5.96, 2.55])
acc['symb32_16_capret_r1_nimg'] = np.array([46.78, 41.90, 37.34, 16.92, 3.10, 0.84, 0.34])
acc['symb32_16_capret_r5_nimg'] = np.array([77.10, 74.24, 68.86, 39.88, 10.48, 3.00, 1.24])
acc['symb32_16_capret_r10_nimg'] = np.array([86.94, 85.20, 80.26, 51.92, 16.94, 4.94, 2.26])

acc['symb32_16_imgret_r1_ncap'] = np.array([36.69, 32.60, 22.82, 7.24, 1.46, 0.41, 0.22])
acc['symb32_16_imgret_r5_ncap'] = np.array([69.20, 66.68, 52.11, 23.24, 5.64, 1.92, 0.91])
acc['symb32_16_imgret_r10_ncap'] = np.array([80.55, 78.26, 65.87, 34.81, 10.35, 3.72, 1.98])
acc['symb32_16_capret_r1_ncap'] = np.array([40.40, 32.48, 17.64, 5.56, 1.38, 0.56, 0.28])
acc['symb32_16_capret_r5_ncap'] = np.array([71.40, 59.98, 36.94, 12.64, 3.08, 1.12, 0.60])
acc['symb32_16_capret_r10_ncap'] = np.array([82.60, 73.14, 49.80, 19.14, 5.18, 1.84, 1.02])
################################################################ 16 256
acc['symb16_256_imgret_r1_both'] = np.array([38.88] * 7)
acc['symb16_256_imgret_r5_both'] = np.array([72.08] * 7)
acc['symb16_256_imgret_r10_both'] = np.array([82.74] * 7)
acc['symb16_256_capret_r1_both'] = np.array([52.00] * 7)
acc['symb16_256_capret_r5_both'] = np.array([80.90] * 7)
acc['symb16_256_capret_r10_both'] = np.array([90.00] * 7)

acc['symb16_256_imgret_r1_nimg'] = np.array([35.92, 29.78, 17.31, 6.63, 1.58, 0.53, 0.24])
acc['symb16_256_imgret_r5_nimg'] = np.array([69.86, 63.31, 44.47, 21.10, 6.38, 2.06, 1.13])
acc['symb16_256_imgret_r10_nimg'] = np.array([81.46, 76.23, 58.91, 31.71, 10.69, 3.89, 2.16])
acc['symb16_256_capret_r1_nimg'] = np.array([48.46, 44.56, 29.92, 9.84, 1.82, 0.46, 0.22])
acc['symb16_256_capret_r5_nimg'] = np.array([79.40, 76.46, 58.24, 25.74, 6.20, 1.66, 0.88])
acc['symb16_256_capret_r10_nimg'] = np.array([88.90, 86.06, 70.16, 37.06, 10.00, 2.98, 1.46])

acc['symb16_256_imgret_r1_ncap'] = np.array([38.92, 37.03, 35.63, 31.89, 20.95, 6.05, 0.73])
acc['symb16_256_imgret_r5_ncap'] = np.array([71.89, 71.49, 70.26, 65.80, 49.77, 20.48, 3.57])
acc['symb16_256_imgret_r10_ncap'] = np.array([82.46, 81.86, 81.12, 78.14, 63.68, 31.32, 7.01])
acc['symb16_256_capret_r1_ncap'] = np.array([47.98, 45.62, 41.48, 31.64, 17.84, 5.14, 1.04])
acc['symb16_256_capret_r5_ncap'] = np.array([79.72, 76.26, 72.24, 58.54, 35.98, 11.34, 2.22])
acc['symb16_256_capret_r10_ncap'] = np.array([88.88, 86.28, 83.62, 72.66, 48.58, 16.78, 3.72])
################################################################ 16 128
acc['symb16_128_imgret_r1_both'] = np.array([38.22] * 7)
acc['symb16_128_imgret_r5_both'] = np.array([72.44] * 7)
acc['symb16_128_imgret_r10_both'] = np.array([83.60] * 7)
acc['symb16_128_capret_r1_both'] = np.array([49.60] * 7)
acc['symb16_128_capret_r5_both'] = np.array([81.40] * 7)
acc['symb16_128_capret_r10_both'] = np.array([89.30] * 7)

acc['symb16_128_imgret_r1_nimg'] = np.array([34.70, 29.49, 17.52, 5.96, 1.68, 0.45, 0.24])
acc['symb16_128_imgret_r5_nimg'] = np.array([69.50, 62.06, 44.66, 19.16, 6.45, 2.11, 1.14])
acc['symb16_128_imgret_r10_nimg'] = np.array([81.09, 75.24, 59.14, 29.64, 10.85, 3.94, 2.14])
acc['symb16_128_capret_r1_nimg'] = np.array([47.92, 43.78, 29.46, 9.20, 1.88, 0.42, 0.20])
acc['symb16_128_capret_r5_nimg'] = np.array([78.26, 74.44, 58.70, 24.76, 6.04, 1.48, 1.06])
acc['symb16_128_capret_r10_nimg'] = np.array([87.94, 84.76, 71.52, 34.96, 10.04, 2.86, 1.72])

acc['symb16_128_imgret_r1_ncap'] = np.array([38.47, 37.49, 35.14, 27.65, 14.18, 2.59, 0.52])
acc['symb16_128_imgret_r5_ncap'] = np.array([70.77, 72.31, 68.13, 60.57, 38.72, 10.42, 2.44])
acc['symb16_128_imgret_r10_ncap'] = np.array([82.23, 82.90, 79.84, 73.88, 52.44, 17.67, 4.62])
acc['symb16_128_capret_r1_ncap'] = np.array([47.22, 45.34, 36.96, 25.20, 11.64, 2.58, 0.84])
acc['symb16_128_capret_r5_ncap'] = np.array([78.58, 74.98, 65.82, 48.58, 24.40, 5.72, 1.42])
acc['symb16_128_capret_r10_ncap'] = np.array([87.58, 85.14, 78.42, 61.80, 34.32, 9.12, 2.48])
################################################################ 16 64
acc['symb16_64_imgret_r1_both'] = np.array([37.98] * 7)
acc['symb16_64_imgret_r5_both'] = np.array([71.92] * 7)
acc['symb16_64_imgret_r10_both'] = np.array([82.24] * 7)
acc['symb16_64_capret_r1_both'] = np.array([50.30] * 7)
acc['symb16_64_capret_r5_both'] = np.array([78.50] * 7)
acc['symb16_64_capret_r10_both'] = np.array([88.60] * 7)

acc['symb16_64_imgret_r1_nimg'] = np.array([35.57, 28.65, 17.69, 6.49, 1.61, 0.39, 0.25])
acc['symb16_64_imgret_r5_nimg'] = np.array([69.22, 61.98, 44.82, 19.75, 6.14, 2.18, 1.06])
acc['symb16_64_imgret_r10_nimg'] = np.array([81.25, 74.96, 59.16, 29.64, 10.61, 3.94, 2.04])
acc['symb16_64_capret_r1_nimg'] = np.array([47.42, 43.68, 29.36, 9.68, 1.48, 0.42, 0.18])
acc['symb16_64_capret_r5_nimg'] = np.array([79.98, 73.38, 58.90, 24.22, 5.60, 1.48, 1.00])
acc['symb16_64_capret_r10_nimg'] = np.array([88.66, 84.32, 71.58, 34.50, 9.18, 2.82, 1.98])

acc['symb16_64_imgret_r1_ncap'] = np.array([37.87, 36.03, 32.72, 22.05, 7.42, 1.06, 0.40])
acc['symb16_64_imgret_r5_ncap'] = np.array([72.60, 70.47, 66.13, 51.37, 22.84, 5.22, 1.85])
acc['symb16_64_imgret_r10_ncap'] = np.array([83.26, 81.17, 78.36, 65.28, 34.03, 9.19, 3.54])
acc['symb16_64_capret_r1_ncap'] = np.array([46.48, 41.96, 32.32, 17.86, 5.56, 1.04, 0.40])
acc['symb16_64_capret_r5_ncap'] = np.array([77.96, 71.30, 59.16, 36.16, 12.92, 2.18, 0.86])
acc['symb16_64_capret_r10_ncap'] = np.array([87.68, 82.74, 72.64, 48.32, 19.62, 4.74, 1.36])
################################################################ 16 32
acc['symb16_32_imgret_r1_both'] = np.array([37.30] * 7)
acc['symb16_32_imgret_r5_both'] = np.array([71.50] * 7)
acc['symb16_32_imgret_r10_both'] = np.array([83.30] * 7)
acc['symb16_32_capret_r1_both'] = np.array([48.90] * 7)
acc['symb16_32_capret_r5_both'] = np.array([78.50] * 7)
acc['symb16_32_capret_r10_both'] = np.array([89.10] * 7)

acc['symb16_32_imgret_r1_nimg'] = np.array([35.46, 28.21, 16.37, 6.02, 1.46, 0.38, 0.16])
acc['symb16_32_imgret_r5_nimg'] = np.array([69.34, 61.52, 43.76, 19.75, 5.85, 2.04, 1.07])
acc['symb16_32_imgret_r10_nimg'] = np.array([80.70, 75.49, 58.58, 30.56, 10.22, 3.84, 2.12])
acc['symb16_32_capret_r1_nimg'] = np.array([47.78, 42.66, 27.68, 9.60, 1.40, 0.26, 0.14])
acc['symb16_32_capret_r5_nimg'] = np.array([78.42, 74.42, 56.78, 24.80, 5.96, 1.46, 0.60])
acc['symb16_32_capret_r10_nimg'] = np.array([88.52, 85.28, 69.50, 35.30, 10.00, 2.82, 1.16])

acc['symb16_32_imgret_r1_ncap'] = np.array([37.06, 33.62, 27.85, 14.32, 2.74, 0.62, 0.27])
acc['symb16_32_imgret_r5_ncap'] = np.array([70.80, 67.80, 61.16, 38.66, 11.19, 2.94, 1.16])
acc['symb16_32_imgret_r10_ncap'] = np.array([81.58, 79.45, 74.34, 52.21, 18.62, 5.44, 2.30])
acc['symb16_32_capret_r1_ncap'] = np.array([44.66, 36.60, 25.86, 10.80, 2.40, 0.84, 0.44])
acc['symb16_32_capret_r5_ncap'] = np.array([74.70, 65.24, 49.22, 23.60, 5.58, 1.64, 0.76])
acc['symb16_32_capret_r10_ncap'] = np.array([84.84, 78.04, 63.02, 34.04, 9.64, 2.86, 1.36])
################################################################ 16 16
acc['symb16_16_imgret_r1_both'] = np.array([33.76] * 7)
acc['symb16_16_imgret_r5_both'] = np.array([68.26] * 7)
acc['symb16_16_imgret_r10_both'] = np.array([80.60] * 7)
acc['symb16_16_capret_r1_both'] = np.array([45.00] * 7)
acc['symb16_16_capret_r5_both'] = np.array([75.50] * 7)
acc['symb16_16_capret_r10_both'] = np.array([85.30] * 7)

acc['symb16_16_imgret_r1_nimg'] = np.array([32.34, 26.44, 15.13, 5.19, 1.34, 0.42, 0.18])
acc['symb16_16_imgret_r5_nimg'] = np.array([66.77, 60.15, 41.73, 17.13, 5.41, 1.98, 1.01])
acc['symb16_16_imgret_r10_nimg'] = np.array([79.26, 73.87, 57.02, 27.14, 9.65, 3.72, 1.81])
acc['symb16_16_capret_r1_nimg'] = np.array([43.76, 39.28, 25.32, 7.66, 1.26, 0.54, 0.28])
acc['symb16_16_capret_r5_nimg'] = np.array([75.20, 71.02, 53.42, 20.84, 5.28, 1.40, 0.98])
acc['symb16_16_capret_r10_nimg'] = np.array([85.36, 81.52, 67.28, 31.02, 8.66, 2.68, 1.64])

acc['symb16_16_imgret_r1_ncap'] = np.array([33.16, 30.37, 20.60, 5.46, 1.24, 0.23, 0.19])
acc['symb16_16_imgret_r5_ncap'] = np.array([67.47, 63.82, 50.71, 19.60, 5.94, 1.74, 0.79])
acc['symb16_16_imgret_r10_ncap'] = np.array([79.92, 76.64, 64.40, 30.18, 10.16, 3.54, 1.78])
acc['symb16_16_capret_r1_ncap'] = np.array([39.04, 28.34, 17.72, 4.92, 1.42, 0.28, 0.24])
acc['symb16_16_capret_r5_ncap'] = np.array([69.72, 55.50, 36.44, 10.66, 3.24, 0.86, 0.56])
acc['symb16_16_capret_r10_ncap'] = np.array([81.36, 69.14, 47.84, 16.72, 5.36, 1.62, 1.00])
################################################################### 256 256 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_imgret_r1'] = [[67.52, 66.66, 64.24, 54.85, 30.05, 6.60, 0.88],  #18
                               [66.07, 65.82, 63.10, 52.77, 26.55, 6.43, 0.92],  #12
                               [62.94, 61.96, 58.55, 48.78, 24.92, 5.41, 1.17],  #6
                               [49.80, 49.03, 46.45, 37.16, 20.20, 5.36, 1.08],  #0
                               [23.86, 22.68, 21.44, 18.48, 10.72, 3.69, 0.78],  #-6
                               [6.19, 5.71, 5.18, 4.90, 3.06, 1.72, 0.64],  #-12
                               [1.16, 1.07, 1.16, 0.95, 0.84, 0.77, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_imgret_r5'] = [[89.80, 89.38, 88.05, 82.67, 59.62, 20.63, 3.48],  #18
                               [89.12, 88.91, 87.66, 81.50, 57.05, 19.87, 4.24],  #12
                               [87.48, 87.46, 85.33, 78.48, 54.29, 18.83,  4.89],  #6
                               [80.58, 78.91, 76.88, 68.87, 47.54, 18.68, 4.51],  #0
                               [54.03, 51.78, 50.07, 45.74, 31.01, 13.54, 3.44],  #-6
                               [19.56, 18.25, 18.14, 17.46, 11.84, 6.74, 3.42],  #-12
                               [4.49, 4.74, 4.93, 4.36, 3.78, 3.06, 1.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_imgret_r10'] = [[94.34, 94.16, 93.30, 89.62, 71.94, 31.25, 6.14],  #18
                               [94.02, 93.85, 93.16, 88.89, 70.18, 29.92, 7.52],  #12
                               [92.90, 92.95, 91.56, 87.25, 67.49, 28.98, 8.52],  #6
                               [88.84, 87.64, 85.71, 79.93, 61.54, 28.66, 8.12],  #0
                               [67.46, 65.13, 64.11, 59.48, 44.16, 21.89, 6.48],  #-6
                               [29.24, 27.91, 28.49, 26.78, 19.34, 11.90, 6.26],  #-12
                               [7.72, 8.24, 8.62, 7.77, 6.85, 5.38, 1.91]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_capret_r1'] = [[81.84, 80.44, 75.24, 57.22, 27.16, 6.66, 1.06],  #18
                               [82.04, 79.54, 73.98, 56.72, 24.84, 5.92, 1.34],  #12
                               [78.98, 77.88, 70.40, 52.44, 23.28, 5.78, 1.46],  #6
                               [68.72, 65.60, 58.84, 43.28, 19.46, 5.26, 1.48],  #0
                               [35.76, 32.90, 29.50, 23.16, 11.84, 3.86, 0.78],  #-6
                               [7.58, 6.58, 6.14, 6.12, 3.66, 1.50, 0.74],  #-12
                               [0.96, 1.18, 1.08, 0.74, 0.76, 0.66, 0.32]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_capret_r5'] = [[95.60, 94.86, 92.58, 80.74,  48.40, 13.02, 1.84],  #18
                               [95.52, 94.84, 92.10, 80.64, 44.62, 12.30, 2.44],  #12
                               [94.56, 93.58, 90.30, 76.94, 42.72, 12.02, 2.76],  #6
                               [90.38, 88.52, 82.64, 68.06, 37.62, 11.60, 2.72],  #0
                               [64.18, 59.00, 54.82, 44.98, 25.34, 8.84, 1.62],  #-6
                               [18.88, 16.72, 16.60, 14.98, 8.60, 4.14, 1.68],  #-12
                               [3.10, 3.06, 3.42, 2.40, 2.18, 1.54, 0.54]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_256_capret_r10'] = [[98.16, 97.70, 96.56, 89.40, 59.90, 19.14, 2.88],  #18
                               [98.06, 97.66, 96.06, 88.48, 57.24, 18.38, 4.08],  #12
                               [97.48, 97.04, 95.72, 85.84, 54.92, 17.98, 4.38],  #6
                               [95.24, 93.94, 90.36, 79.82, 49.90, 17.26, 4.44],  #0
                               [76.56, 71.72, 68.00, 57.70, 35.40, 13.42, 2.84],  #-6
                               [26.76, 24.36, 25.84, 22.66, 13.54, 7.24, 3.02],  #-12
                               [4.74, 4.88, 5.62, 4.24, 3.60, 2.70, 0.98]]   #-18
################################################################### 256 128 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_imgret_r1'] = [[66.19, 64.76, 60.46, 44.66, 16.34, 2.64, 0.40],  #18
                               [64.90, 63.32, 59.25, 42.84, 15.72, 3.10, 0.44],  #12
                               [61.00, 59.22, 55.09, 40.24, 15.37, 2.90, 0.63],  #6
                               [48.52, 46.36, 42.36, 31.38, 12.28, 2.85, 0.41],  #0
                               [23.32, 21.84, 19.66, 15.47, 7.56, 2.00, 0.58],  #-6
                               [5.62, 6.12, 5.97, 4.39, 2.52, 0.94, 0.29],  #-12
                               [1.05, 1.22, 1.05, 1.07, 0.77, 0.32, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_imgret_r5'] = [[88.96, 88.29, 86.19, 75.22, 41.55, 9.86, 1.59],  #18
                               [88.41, 87.64, 85.65, 73.50, 40.95, 11.59, 2.32],  #12
                               [87.14, 86.10, 83.23, 71.42, 41.16, 11.80, 2.96],  #6
                               [79.11, 77.78, 73.78, 62.67, 35.12, 10.31, 2.40],  #0
                               [53.01, 50.45, 47.42, 40.35, 23.51, 7.69, 2.24],  #-6
                               [18.03, 18.94, 18.61, 14.83, 9.33, 4.31, 1.30],  #-12
                               [3.87, 4.70, 4.56, 4.34, 3.05, 1.74, 0.93]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_imgret_r10'] = [[94.04, 93.59, 92.24, 84.11, 54.87, 16.04, 3.47],  #18
                               [93.54, 93.21, 91.50, 83.19, 54.00, 18.89, 4.50],  #12
                               [92.84, 92.17, 90.25, 81.66, 54.86, 19.42, 5.56],  #6
                               [87.68, 86.85, 83.80, 75.00, 48.63, 16.90, 4.88],  #0
                               [66.61, 64.06, 60.92, 54.07, 34.90, 13.64, 4.14],  #-6
                               [27.31, 28.84, 27.93, 23.57, 15.86, 7.56, 3.01],  #-12
                               [6.94, 8.31, 8.36, 7.64, 5.95, 3.26, 1.86]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_capret_r1'] = [[80.26, 77.16, 67.74, 43.80, 13.54, 2.44, 0.58],  #18
                               [79.42, 76.20, 66.82, 41.88, 13.64, 3.02, 0.50],  #12
                               [77.06, 73.06, 63.60, 40.44, 13.46, 2.84, 0.98],  #6
                               [66.72, 61.86, 51.30, 31.96, 10.92, 2.70, 0.56],  #0
                               [34.38, 29.24, 25.28, 17.24, 7.50, 1.96, 0.54],  #-6
                               [5.70, 6.80, 6.32, 4.26, 2.64, 1.18, 0.24],  #-12
                               [0.90, 1.12, 1.04, 0.90, 0.64, 0.36, 0.10]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_capret_r5'] = [[95.04, 93.58, 88.64, 67.56, 27.14, 5.28, 0.96],  #18
                               [94.76, 93.32, 88.52, 65.56, 25.98, 6.52, 1.16],  #12
                               [94.10, 92.14, 86.16, 65.96, 26.12, 6.32, 1.62],  #6
                               [88.46, 85.40, 76.38, 57.38, 23.50, 5.40, 1.34],  #0
                               [61.88, 55.26, 49.14, 36.68, 16.72, 4.50, 1.16],  #-6
                               [15.94, 17.28, 16.66, 11.66, 6.96, 2.60, 0.70],  #-12
                               [2.48, 3.20, 2.98, 3.02, 1.84, 1.08, 0.36]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_128_capret_r10'] = [[97.68, 96.88, 94.08, 78.68, 37.92, 8.60, 1.60],  #18
                               [97.72, 96.78, 94.12, 77.66, 36.76, 10.02, 2.12],  #12
                               [97.22, 96.38, 92.82, 76.10, 36.04, 10.16, 2.88],  #6
                               [94.12, 92.18, 85.82, 70.04, 33.22, 8.62, 2.36],  #0
                               [73.96, 68.88, 62.58, 47.96, 24.20, 7.26, 1.96],  #-6
                               [23.80, 25.46, 24.62, 17.92, 11.32, 4.34, 1.30],  #-12
                               [4.26, 5.30, 5.36, 5.24, 3.12, 1.60, 0.74]]   #-18
################################################################### 256 64 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_imgret_r1'] = [[62.90, 61.31, 52.75, 31.90, 8.13, 1.48, 0.26],  #18
                               [62.80, 59.88, 51.64, 31.18, 8.42, 1.74, 0.32],  #12
                               [58.52, 56.12, 47.82, 28.78, 8.08, 1.68, 0.40],  #6
                               [46.57, 44.20, 36.78, 22.35, 6.12, 1.40, 0.38],  #0
                               [22.33, 20.42, 17.86, 11.13, 4.07, 1.02, 0.24],  #-6
                               [5.06, 5.21, 4.35, 3.08, 1.92, 0.72, 0.21],  #-12
                               [0.95, 1.09, 1.05, 0.64, 0.48, 0.22, 0.17]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_imgret_r5'] = [[88.01, 86.42, 81.55, 62.95, 25.52, 5.49, 1.49],  #18
                               [87.60, 86.12, 80.71, 62.00, 25.51, 6.78, 1.88],  #12
                               [85.50, 84.20, 77.92, 59.72, 24.72, 6.40, 1.85],  #6
                               [77.96, 75.86, 68.97, 51.56, 21.20, 6.18, 1.88],  #0
                               [51.15, 48.50, 44.62, 32.23, 14.40,  4.60, 1.49],  #-6
                               [16.81, 17.25, 15.59, 11.77, 7.43, 2.94, 1.12],  #-12
                               [3.98, 4.58, 4.50, 3.12, 2.31, 1.15, 0.80]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_imgret_r10'] = [[93.29, 92.20, 89.11, 74.97, 36.58, 9.42, 2.76],  #18
                               [92.96, 92.01, 88.70, 73.75, 37.24, 11.72, 3.68],  #12
                               [91.94, 90.94, 86.64, 71.95, 36.71, 11.19, 3.81],  #6
                               [86.96, 85.29, 80.16, 65.56, 31.61, 10.66, 3.36],  #0
                               [65.06, 62.45, 58.54, 46.04, 23.24, 8.20, 3.12],  #-6
                               [25.59, 26.52, 24.96, 19.91, 12.64, 5.51, 2.14],  #-12
                               [7.05, 8.46, 8.07, 6.05, 4.07, 2.17, 1.80]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_capret_r1'] = [[77.04, 70.66, 54.64, 28.12, 7.22, 1.44, 0.42],  #18
                               [76.40, 71.30, 54.72, 26.94, 6.74, 1.58, 0.52],  #12
                               [74.82, 68.08, 52.96, 25.88, 6.96, 1.52, 0.54],  #6
                               [64.16, 57.36, 41.58, 20.98, 5.36, 1.58, 0.52],  #0
                               [32.34, 27.82, 22.32, 11.90, 3.80, 1.10, 0.34],  #-6
                               [5.52, 5.74, 5.10, 3.30, 1.90, 0.62, 0.06],  #-12
                               [0.78, 0.94, 0.86, 0.74, 0.42, 0.20, 0.18]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_capret_r5'] = [[93.76, 90.38, 79.22, 49.74, 15.08, 2.88, 0.70],  #18
                               [93.58, 90.44, 79.08, 48.14, 14.12, 3.50, 0.94],  #12
                               [92.34, 88.54, 76.96, 47.16, 14.66, 3.66, 1.06],  #6
                               [87.20, 82.22, 68.24, 40.74, 12.54, 3.08, 1.02],  #0
                               [60.00, 53.44, 44.46, 26.46, 8.90, 2.32, 0.64],  #-6
                               [14.60, 15.06, 13.44, 9.38, 4.64, 1.58, 0.34],  #-12
                               [2.68, 3.24, 3.02, 2.30, 1.20, 0.52, 0.46]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_64_capret_r10'] = [[97.24, 95.48, 88.64, 61.94, 21.56, 4.96, 1.30],  #18
                               [96.94, 95.54, 88.62, 61.50, 21.02, 5.48, 1.42],  #12
                               [96.20, 94.40, 86.12, 59.26, 21.66, 6.04, 1.58],  #6
                               [93.30, 90.46, 79.44, 53.92, 18.72, 5.10, 1.84],  #0
                               [72.98, 66.20, 57.74, 37.44, 13.64, 4.36, 1.16],  #-6
                               [21.56, 23.00, 20.36, 14.34, 7.40, 2.74, 0.60],  #-12
                               [4.14, 5.98, 4.74, 4.02, 1.98, 1.06, 0.96]]   #-18
################################################################### 256 32 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_imgret_r1'] = [[53.94, 51.33, 40.63, 17.54, 3.66, 0.76, 0.21],  #18
                               [53.81, 49.81, 39.11, 17.18, 3.42, 0.71, 0.23],  #12
                               [50.13, 47.03, 36.58, 16.72, 3.28, 0.85, 0.18],  #6
                               [39.40, 37.05, 29.00, 12.82, 3.16, 0.46, 0.24],  #0
                               [20.60, 18.74, 14.07, 7.58, 2.12, 0.64, 0.23],  #-6
                               [5.42, 4.91, 4.11, 2.12, 0.96, 0.36, 0.16],  #-12
                               [0.89, 1.20, 0.91, 0.45, 0.20, 0.06, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_imgret_r5'] = [[83.07, 80.68, 72.76, 43.60, 13.76, 3.50, 1.45],  #18
                               [82.67, 80.08, 71.93, 43.18, 12.34, 3.52, 1.25],  #12
                               [80.66, 78.88, 69.02, 42.36, 12.38, 3.36, 1.04],  #6
                               [72.78, 70.23, 60.21, 35.60, 11.86, 2.89, 1.36],  #0
                               [49.17, 45.87, 37.56, 23.29, 7.66, 2.75, 0.98],  #-6
                               [16.92, 16.23, 14.27, 8.79, 4.68, 1.84, 0.93],  #-12
                               [3.81, 4.32, 3.77, 2.41, 1.20, 0.87, 0.50]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_imgret_r10'] = [[90.25, 88.55, 82.40, 57.34, 22.08, 6.52, 2.78],  #18
                               [90.01, 88.23, 82.01, 57.04, 20.27, 6.53, 2.67],  #12
                               [88.85, 87.27, 80.02, 55.87, 20.61, 6.03, 2.25],  #6
                               [83.25, 81.42, 72.99, 49.19, 19.30, 5.50, 2.42],  #0
                               [64.02, 60.60, 51.68, 34.56, 13.14, 4.86, 2.11],  #-6
                               [25.77, 25.73, 22.38, 15.24, 8.60, 3.53, 1.70],  #-12
                               [6.90, 7.75, 6.68, 4.72, 2.20, 1.81, 1.02]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_capret_r1'] = [[66.34, 59.26, 38.38, 13.92, 3.46, 0.64, 0.38],  #18
                               [66.62, 56.68, 37.62, 14.20, 3.02, 1.10, 0.36],  #12
                               [63.10, 54.82, 35.56, 13.52, 3.04, 0.76, 0.24],  #6
                               [54.82, 46.82, 29.52, 11.26, 3.10, 0.78, 0.28],  #0
                               [31.68, 25.42, 16.54, 7.70, 1.96, 0.72, 0.16],  #-6
                               [5.80, 5.50, 4.96, 2.20, 1.00, 0.38, 0.14],  #-12
                               [0.80, 1.20, 0.64, 0.64, 0.32, 0.14, 0.12]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_capret_r5'] = [[89.26, 82.88, 63.52, 28.20, 6.96, 1.66, 0.78],  #18
                               [89.42, 81.86, 62.80, 28.00, 6.32, 2.16, 0.62],  #12
                               [88.06, 82.14, 61.58, 27.80, 6.74, 1.54, 0.48],  #6
                               [82.22, 73.56, 54.62, 23.14, 7.18, 1.58, 0.54],  #0
                               [58.60, 49.70, 35.12, 16.44, 4.58, 1.36, 0.38],  #-6
                               [14.84, 15.10, 12.26, 6.16, 2.70, 0.96, 0.38],  #-12
                               [2.52, 3.28, 2.24, 1.66, 0.66, 0.40, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_32_capret_r10'] = [[94.50, 90.82, 76.08, 38.96, 10.90, 2.86, 1.22],  #18
                               [94.52, 90.02, 75.60, 38.20, 9.68, 3.48, 1.04],  #12
                               [93.82, 89.82, 74.98, 38.34, 10.98, 2.50, 1.00],  #6
                               [90.44, 84.22, 67.50, 33.08, 10.82, 2.52, 0.84],  #0
                               [71.14, 64.38, 46.88, 24.66, 7.50, 2.36, 0.68],  #-6
                               [22.38, 22.72, 18.62, 9.88, 4.34, 1.50, 0.60],  #-12
                               [4.58, 7.78, 3.94, 2.84, 1.14, 0.68, 0.38]]   #-18
################################################################### 256 16 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_imgret_r1'] = [[37.14, 33.48, 24.06, 8.18, 1.55, 0.52, 0.20],  #18
                               [35.46, 32.44, 22.62, 7.83, 1.47, 0.37, 0.30],  #12
                               [33.81, 31.00, 21.36, 7.29, 1.46, 0.40, 0.16],  #6
                               [27.49, 23.79, 17.48, 6.02, 1.44, 0.36, 0.22],  #0
                               [16.44, 14.14, 10.11, 3.88, 1.08, 0.40, 0.14],  #-6
                               [4.58, 3.97, 2.85, 1.34, 0.59, 0.18, 0.21],  #-12
                               [0.81, 0.92, 0.78, 0.16, 0.14, 0.06, 0.05]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_imgret_r5'] = [[71.51, 67.41, 54.84, 24.43, 6.33, 2.26, 1.15],  #18
                               [68.98, 66.02, 53.45, 24.79, 6.42, 2.04, 1.10],  #12
                               [67.05, 64.14, 50.94, 23.52, 6.44, 1.65, 1.08],  #6
                               [59.94, 56.08, 45.42, 19.86, 5.61, 1.74, 0.91],  #0
                               [42.42, 38.85, 29.95, 14.30, 4.28, 1.41, 0.84],  #-6
                               [15.35, 14.47, 10.69, 6.30, 2.07, 0.94, 0.95],  #-12
                               [3.46, 3.78, 3.14, 1.01, 0.90, 0.44, 0.46]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_imgret_r10'] = [[82.42, 79.40, 68.10, 36.03, 11.35, 4.20, 2.13],  #18
                               [80.89, 78.19, 67.48, 36.83, 10.98, 3.66, 2.03],  #12
                               [78.90, 76.58, 65.03, 35.05, 11.50, 3.41, 2.14],  #6
                               [73.61, 70.06, 60.15, 30.78, 10.18, 3.29, 1.98],  #0
                               [56.52, 53.48, 43.67, 22.64, 8.12, 2.83, 1.55],  #-6
                               [24.41, 23.06, 17.93, 11.34, 3.87, 1.89, 1.71],  #-12
                               [6.20, 6.57, 5.65, 2.14, 1.74, 0.86, 1.02]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_capret_r1'] = [[43.30, 33.00, 19.70, 6.56, 1.54, 0.60, 0.20],  #18
                               [40.60, 33.06, 19.00, 6.10, 1.28, 0.50, 0.32],  #12
                               [40.58, 31.20, 18.26, 6.70, 1.64, 0.34, 0.18],  #6
                               [36.78, 27.32, 16.88, 5.48, 1.42, 0.46, 0.26],  #0
                               [24.00, 19.00, 10.90, 3.76, 0.90, 0.34, 0.26],  #-6
                               [5.40, 4.60, 3.46, 1.70, 0.64, 0.14, 0.22],  #-12
                               [0.76, 0.82, 0.66, 0.24, 0.20, 0.10, 0.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_capret_r5'] = [[74.06, 60.44, 39.32, 13.46, 3.24, 1.14, 0.42],  #18
                               [71.84, 60.64, 38.28, 13.60, 3.20, 1.02, 0.56],  #12
                               [71.12, 60.20, 36.86, 13.40, 3.40, 0.80, 0.36],  #6
                               [65.82, 54.56, 34.90, 11.72, 3.16, 1.00, 0.48],  #0
                               [49.42, 41.26, 24.98, 8.84, 2.30, 0.66, 0.38],  #-6
                               [13.60, 12.56, 8.66, 3.96, 1.40, 0.32, 0.46],  #-12
                               [2.32, 2.84, 1.90, 0.70, 0.46, 0.14, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb256_16_capret_r10'] = [[85.26, 73.90, 52.04, 20.46, 5.40, 2.08, 0.94],  #18
                               [83.28, 73.74, 50.64, 20.32, 5.52, 1.80, 0.84],  #12
                               [81.98, 73.26, 49.52, 19.82, 6.04, 1.26, 0.56],  #6
                               [78.00, 67.78, 46.90, 18.02, 4.98, 1.46, 0.72],  #0
                               [62.94, 54.94, 35.18, 13.92, 3.96, 1.22, 0.74],  #-6
                               [20.84, 19.04, 13.14, 6.80, 2.10, 0.74, 0.94],  #-12
                               [3.64, 4.92, 3.54, 1.32, 0.76, 0.20, 0.50]]   #-18
################################################################### 128 256 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_imgret_r1'] = [[65.92, 65.16, 62.12, 53.48, 27.98, 5.38, 0.94],  #18
                               [63.85, 63.23, 60.48, 50.04, 25.73, 6.20, 1.04],  #12
                               [56.84, 56.30, 52.74, 42.97, 21.86, 5.79, 1.11],  #6
                               [39.08, 37.52, 35.48, 28.76, 15.86, 4.46, 0.94],  #0
                               [13.44, 13.16, 12.72, 10.98, 7.30, 2.70, 0.62],  #-6
                               [2.53, 2.73, 2.53, 2.63, 1.88, 1.04, 0.31],  #-12
                               [0.44, 0.57, 0.61, 0.62, 0.66, 0.38, 0.33]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_imgret_r5'] = [[89.04, 88.92, 87.22, 81.67, 58.02, 17.74, 3.86],  #18
                               [88.13, 88.04, 86.29, 80.07, 55.37, 20.05, 4.78],  #12
                               [84.86, 84.41, 82.12, 74.80, 50.60, 19.62, 4.71],  #6
                               [71.55, 69.66, 67.72, 60.98, 41.06, 15.60, 4.06],  #0
                               [36.16, 35.75, 35.54, 31.27, 22.87, 9.89, 3.31],  #-6
                               [9.61, 10.34, 10.38, 9.93, 8.02, 4.41, 1.43],  #-12
                               [2.28, 2.71, 3.05, 2.80, 2.68, 1.63, 1.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_imgret_r10'] = [[94.13, 93.86, 92.86, 89.08, 70.96, 27.81, 6.54],  #18
                               [93.34, 93.42, 92.08, 88.18, 68.45, 30.29, 8.37],  #12
                               [91.35, 91.00, 89.76, 84.52, 64.46, 30.17, 8.46],  #6
                               [81.92, 80.77, 79.28, 73.78, 54.74, 24.72, 7.35],  #0
                               [49.86, 49.24, 49.15, 44.28, 34.33, 16.88, 5.81],  #-6
                               [16.23, 17.11, 17.36, 16.75, 13.84, 8.06, 3.29],  #-12
                               [4.52, 4.94, 5.70, 5.20, 4.72, 3.08, 2.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_capret_r1'] = [[81.80, 79.82, 72.90, 56.90, 25.68, 5.02, 1.06],  #18
                               [80.14, 78.44, 71.80, 54.16, 23.86, 5.96, 1.16],  #12
                               [75.38, 72.88, 66.52, 48.16, 21.10, 5.86, 1.24],  #6
                               [56.92, 53.76, 47.58, 35.56, 16.32, 4.72, 1.04],  #0
                               [20.86, 19.20, 18.46, 14.78, 7.48, 2.62, 0.70],  #-6
                               [2.38, 3.06, 3.50, 3.00, 2.08, 0.88, 0.54],  #-12
                               [0.44, 0.62, 0.56, 0.60, 0.40, 0.36, 0.36]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_capret_r5'] = [[95.66, 94.78, 91.08, 80.26, 44.54, 11.04, 2.16],  #18
                               [94.82, 94.26, 90.54, 78.50, 43.76, 12.02, 2.30],  #12
                               [93.40, 92.32, 87.88, 73.94, 40.62, 12.64, 2.86],  #6
                               [84.02, 80.12, 74.80, 61.86, 33.00, 9.88, 2.46],  #0
                               [43.64, 41.16, 40.40, 32.92, 18.56, 6.28, 1.82],  #-6
                               [7.76, 8.76, 9.66, 8.10, 5.46, 2.66, 0.94],  #-12
                               [1.26, 1.38, 1.94, 1.82, 1.42, 1.06, 0.68]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_256_capret_r10'] = [[98.22, 97.84, 95.70, 88.70, 57.12, 16.52, 3.56],  #18
                               [97.50, 97.72, 95.66, 87.70, 56.12, 18.18, 4.12],  #12
                               [96.78, 96.22, 93.80, 84.82, 52.18, 18.08, 4.44],  #6
                               [90.86, 88.64, 84.56, 74.04, 44.54, 14.86, 3.74],  #0
                               [56.86, 53.68, 53.74, 44.38, 27.06, 10.16, 3.00],  #-6
                               [12.20, 13.78, 15.18, 12.44, 9.06, 4.24, 1.80],  #-12
                               [2.30, 2.76, 3.30, 3.08, 2.62, 1.66, 1.08]]   #-18
################################################################### 128 128 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_imgret_r1'] = [[64.08, 63.22, 58.22, 43.67, 15.18, 2.68, 0.35],  #18
                               [62.58, 60.91, 56.15, 40.70, 16.70, 3.50, 1.06],  #12
                               [55.90, 54.99, 48.48, 35.51, 14.98, 2.55, 0.75],  #6
                               [36.89, 35.54, 31.10, 24.32, 11.04, 2.25, 0.48],  #0
                               [13.70, 12.13, 12.72, 8.47, 4.70, 1.53, 0.53],  #-6
                               [2.73, 2.70, 2.92, 2.02, 1.40, 0.88, 0.17],  #-12
                               [0.59, 0.68, 0.85, 0.57, 0.38, 0.26, 0.26]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_imgret_r5'] = [[88.44, 87.68, 85.13, 74.35, 39.93, 9.82, 2.07],  #18
                               [87.22, 86.72, 83.97, 71.76, 42.00, 12.20, 3.54],  #12
                               [83.74, 83.90, 79.14, 67.94, 38.64, 10.54, 3.52],  #6
                               [69.53, 67.92, 62.40, 55.07, 31.73, 9.44, 2.26],  #0
                               [36.55, 33.65, 34.52, 26.48, 15.89, 6.26, 2.28],  #-6
                               [9.98, 9.97, 10.56, 8.51, 6.01, 3.10, 1.32],  #-12
                               [2.24, 2.83, 3.12, 2.51, 2.07, 1.21, 0.90]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_imgret_r10'] = [[93.63, 93.22, 91.54, 84.08, 53.63, 16.53, 3.84],  #18
                               [92.95, 92.54, 90.66, 82.30, 55.00, 19.55, 6.37],  #12
                               [90.59, 90.76, 87.62, 79.04, 51.98, 17.81, 6.53],  #6
                               [80.76, 79.60, 75.12, 68.80, 44.66, 15.97, 4.51],  #0
                               [50.18, 47.06, 48.11, 39.08, 25.18, 10.60, 4.16],  #-6
                               [15.78, 16.36, 17.93, 14.87, 10.94, 5.88, 2.63],  #-12
                               [4.19, 5.36, 5.56, 4.76, 3.98, 2.34, 1.74]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_capret_r1'] = [[79.60, 76.18, 66.12, 41.86, 12.90, 2.72, 0.48],  #18
                               [78.88, 75.02, 64.04, 39.78, 13.64, 3.04, 0.92],  #12
                               [73.82, 70.78, 59.20, 35.44, 13.62, 2.84, 0.80],  #6
                               [54.76, 51.36, 39.86, 27.64, 10.46, 2.62, 0.50],  #0
                               [20.54, 17.30, 18.78, 10.62, 4.14, 1.72, 0.44],  #-6
                               [2.44, 2.78, 3.04, 2.58, 1.50, 0.72, 0.28],  #-12
                               [0.30, 0.60, 0.52, 0.56, 0.42, 0.22, 0.20]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_capret_r5'] = [[94.98, 93.74, 87.92, 66.98, 25.72, 5.78, 1.10],  #18
                               [94.10, 92.94, 86.42, 64.86, 27.24, 6.74, 1.74],  #12
                               [92.54, 90.32, 83.04, 60.28, 25.80, 6.20, 1.68],  #6
                               [81.52, 78.66, 66.06, 50.82, 21.58, 5.98, 1.26],  #0
                               [43.94, 38.76, 38.84, 23.44, 10.86, 3.70, 0.92],  #-6
                               [7.50, 8.40, 9.78, 7.52, 4.24, 2.12, 0.56],  #-12
                               [1.30, 1.78, 1.84, 1.68, 1.38, 0.68, 0.44]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_128_capret_r10'] = [[97.56, 97.18, 94.00, 79.66, 36.16, 8.82, 1.94],  #18
                               [97.22, 96.82, 92.96, 76.30, 37.84, 10.76, 2.86],  #12
                               [96.10, 95.28, 90.30, 72.86, 35.42, 9.62, 2.90],  #6
                               [89.64, 87.56, 78.02, 64.16, 30.76, 9.20, 2.26],  #0
                               [56.44, 50.62, 51.34, 33.24, 17.32, 6.24, 1.70],  #-6
                               [12.40, 13.58, 15.46, 11.36, 7.26, 3.52, 1.02],  #-12
                               [2.16, 3.10, 3.16, 2.94, 2.18, 1.16, 0.56]]   #-18
################################################################### 128 64 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb128_64_imgret_r1'] = [[62.22, 59.48, 51.86, 30.23, 8.32, 1.54, 0.31],  #18
                               [60.16, 57.44, 50.05, 28.78, 8.13, 1.49, 0.40],  #12
                               [53.00, 51.09, 43.35, 25.17, 7.41, 1.22, 0.52],  #6
                               [35.09, 33.32, 28.26, 16.54, 5.32, 1.11, 0.37],  #0
                               [12.49, 12.29, 10.41, 6.64, 2.92, 0.86, 0.30],  #-6
                               [2.47, 2.68, 2.52, 1.86, 1.28, 0.49, 0.19],  #-12
                               [0.54, 0.74, 0.60, 0.50, 0.34, 0.10, 0.05]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_64_imgret_r5'] = [[86.63, 85.68, 80.66, 61.38, 24.47, 6.18, 1.29],  #18
                               [86.13, 84.75, 79.97, 59.48, 25.60, 6.72, 1.90],  #12
                               [82.50, 80.54, 74.56, 55.64, 23.07, 5.80, 1.88],  #6
                               [67.91, 65.39, 59.90, 43.24, 18.64, 4.96, 1.80],  #0
                               [34.53, 34.18, 30.02, 22.30, 10.76, 3.54, 1.22],  #-6
                               [9.99, 10.08, 9.34, 7.11, 4.59, 1.95, 1.05],  #-12
                               [2.52, 2.68, 2.74, 2.66, 1.58, 0.87, 0.46]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_64_imgret_r10'] = [[92.46, 92.02, 88.58, 73.77, 35.78, 10.41, 2.53],  #18
                               [92.39, 91.37, 87.72, 72.44, 37.18, 11.44, 3.56],  #12
                               [90.01, 88.46, 84.32, 68.65, 34.12, 10.30, 3.48],  #6
                               [79.56, 77.62, 72.97, 57.38, 28.64, 8.91, 3.40],  #0
                               [48.15, 47.57, 43.02, 34.02, 17.67, 6.52, 2.34],  #-6
                               [16.49, 16.66, 15.81, 12.16, 8.21, 3.68, 2.17],  #-12
                               [4.56, 5.25, 5.28, 4.64, 3.00, 2.05, 0.86]]   #-18
################################  18    12    6      0     -6    -12   -18
acc['symb128_64_capret_r1'] = [[74.64, 70.96, 54.24, 26.00, 6.54, 1.26, 0.30],  #18
                               [75.84, 69.24, 52.58, 25.12, 6.34, 1.74, 0.42],  #12
                               [69.78, 63.60, 47.36, 22.80, 6.00, 1.62, 0.54],  #6
                               [51.46, 45.14, 34.52, 17.54, 5.10, 1.08, 0.38],  #0
                               [18.78, 17.48, 13.40, 7.60, 2.72, 0.82, 0.18],  #-6
                               [2.86, 2.70, 3.08, 1.86, 0.78, 0.38, 0.28],  #-12
                               [0.46, 0.44, 0.68, 0.50, 0.30, 0.30, 0.06]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_64_capret_r5'] = [[93.24, 90.64, 77.38, 48.10, 13.42, 2.98, 0.66],  #18
                               [93.38, 89.92, 77.58, 45.28, 13.92, 3.40, 0.90],  #12
                               [90.90, 86.18, 73.16, 42.88, 13.16, 3.30, 0.96],  #6
                               [79.10, 73.70, 60.30, 34.34, 11.40, 2.54, 0.70],  #0
                               [40.62, 38.06, 30.62, 18.54, 6.80, 2.08, 0.46],  #-6
                               [8.14, 8.54, 8.36, 5.32, 2.78, 0.84,  0.56],  #-12
                               [1.46, 1.80, 1.90, 1.56, 0.92, 0.52, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_64_capret_r10'] = [[97.10, 95.48, 87.16, 60.32, 19.84, 4.84, 1.20],  #18
                               [96.82, 95.10, 86.78, 58.70, 20.84, 5.64, 1.66],  #12
                               [95.52, 92.52, 83.38, 56.02, 19.64, 5.02, 1.60],  #6
                               [88.26, 83.94, 72.86, 46.44, 17.12, 4.10, 1.22],  #0
                               [52.54, 50.74, 42.56, 27.30, 10.86, 3.32, 0.90],  #-6
                               [12.58, 14.00, 12.90, 8.50, 4.92, 1.66, 0.96],  #-12
                               [2.60, 3.08, 3.12, 2.74, 1.46, 1.10, 0.36]]   #-18
################################################################### 128 32 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_imgret_r1'] = [[53.14, 50.26, 40.14, 16.96, 3.32, 0.70, 0.27],  #18
                               [51.78, 48.89, 38.78, 16.86, 3.52, 0.82, 0.25],  #12
                               [45.88, 43.86, 33.82, 14.61, 3.28, 0.76, 0.32],  #6
                               [31.18, 29.58, 21.94, 9.35, 2.32, 0.76, 0.15],  #0
                               [11.70, 11.12, 9.02, 4.78, 1.58, 0.60, 0.13],  #-6
                               [2.53, 2.29, 2.42, 1.68, 0.83, 0.24, 0.20],  #-12
                               [0.51, 0.68, 0.63, 0.49, 0.26, 0.25, 0.04]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_imgret_r5'] = [[82.50, 80.32, 71.93, 43.38, 12.13, 3.25, 1.37],  #18
                               [81.78, 79.26, 70.66, 42.63, 12.48, 3.43, 1.50],  #12
                               [77.87, 74.76, 65.56, 39.07, 12.06, 3.09, 1.24],  #6
                               [64.46, 62.59, 51.79, 28.93, 9.74, 2.99, 0.98],  #0
                               [32.87, 31.78, 26.98, 16.08, 6.83, 2.38, 0.93],  #-6
                               [9.43, 9.46, 8.73, 5.66, 3.59, 1.06, 0.79],  #-12
                               [2.46, 2.91, 2.93, 2.46, 1.07, 0.94, 0.50]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_imgret_r10'] = [[89.94, 88.32, 81.98, 56.70, 20.59, 5.76, 2.66],  #18
                               [89.41, 87.68, 81.18, 56.28, 20.17, 6.22, 2.61],  #12
                               [86.91, 84.53, 77.50, 52.93, 19.67, 5.72, 2.73],  #6
                               [77.16, 75.28, 65.81, 41.90, 16.46, 5.45, 2.07],  #0
                               [46.37, 44.74,  39.58, 25.54, 11.70, 4.54, 1.86],  #-6
                               [15.40, 15.73, 14.27, 10.13, 6.90, 2.14, 1.64],  #-12
                               [4.40, 5.34, 5.24, 4.34, 1.97, 2.00, 1.05]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_capret_r1'] = [[64.54, 56.76, 37.88, 14.06, 2.90, 1.02, 0.32],  #18
                               [64.04, 55.82, 36.86, 13.12, 2.94, 0.68, 0.44],  #12
                               [60.94, 52.14, 34.64, 11.76, 3.06, 0.76, 0.24],  #6
                               [47.24, 40.24, 25.08, 8.94, 2.36, 0.72, 0.44],  #0
                               [16.14, 14.82, 10.62, 4.80, 1.68, 0.52, 0.20],  #-6
                               [2.38, 2.76, 2.10, 1.34, 0.82, 0.24, 0.26],  #-12
                               [0.52, 0.62, 0.52, 0.50, 0.12, 0.22, 0.06]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_capret_r5'] = [[87.98, 82.46, 63.80, 27.38, 6.54, 1.76, 0.74],  #18
                               [88.22, 81.62, 63.16, 27.32, 6.40, 1.68, 0.90],  #12
                               [86.02, 77.82, 58.34, 25.26, 6.84, 1.50, 0.72],  #6
                               [74.48, 67.86, 47.94, 19.66, 5.36, 1.38, 0.70],  #0
                               [38.24, 34.66, 24.88, 11.02, 4.06, 1.02, 0.48],  #-6
                               [7.62, 8.44, 6.34, 4.18, 2.18, 0.64, 0.34],  #-12
                               [1.54, 2.04, 1.66, 1.44, 0.64, 0.54, 0.18]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_32_capret_r10'] = [[93.54, 91.24, 75.98, 37.74, 10.64, 2.94, 1.14],  #18
                               [94.42, 90.12, 75.78, 38.30, 10.70, 2.66, 1.30],  #12
                               [92.60, 86.94, 70.56, 35.46, 11.08, 2.52, 1.22],  #6
                               [84.66, 80.22, 60.66, 29.12, 9.08, 2.86, 1.32],  #0
                               [50.88, 47.70, 35.40, 17.08, 6.80, 1.98, 0.86],  #-6
                               [11.70, 13.42, 10.40, 6.92, 3.80, 1.04, 0.62],  #-12
                               [2.48, 3.68, 3.26, 2.58, 1.12, 0.84, 0.42]]   #-18
################################################################### 128 16 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_imgret_r1'] = [[37.42, 33.47, 22.85, 7.44, 1.40, 0.52, 0.18],  #18
                               [34.31, 31.36, 22.22, 7.74, 1.37, 0.36, 0.16],  #12
                               [30.87, 28.61, 19.37, 7.25, 1.62, 0.42, 0.27],  #6
                               [21.98, 20.22, 14.59, 5.50, 1.17, 0.38, 0.18],  #0
                               [9.98, 8.85, 6.57, 2.47, 0.97, 0.28, 0.28],  #-6
                               [2.64, 2.42, 1.78, 1.10, 0.46, 0.18, 0.11],  #-12
                               [0.58, 0.59, 0.48, 0.27, 0.16, 0.10, 0.09]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_imgret_r5'] = [[70.80, 66.42, 52.98, 24.73, 6.60, 2.26, 0.93],  #18
                               [68.60, 64.61, 52.73, 24.88, 6.41, 1.74, 0.96],  #12
                               [64.34, 60.79, 48.92, 22.10, 6.23, 1.76, 1.03],  #6
                               [52.95, 50.92, 39.78, 18.40, 5.01, 1.76, 0.71],  #0
                               [29.74, 27.45, 21.38, 9.77, 4.29, 1.29, 0.79],  #-6
                               [9.10, 9.14, 6.92, 4.72, 1.95, 0.87, 0.59],  #-12
                               [2.78, 2.82, 2.39, 1.68, 1.03, 0.56, 0.52]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_imgret_r10'] = [[81.74, 78.58, 67.06, 36.65, 11.51, 4.15, 1.90],  #18
                               [80.54, 77.40, 66.41, 36.52, 11.38, 3.44, 1.85],  #12
                               [77.28, 74.13, 63.12, 33.51, 10.82, 3.61, 1.86],  #6
                               [67.32, 65.18, 54.57, 28.26, 9.14, 3.56, 1.46],  #0
                               [43.06, 40.28, 32.83, 16.84, 7.44, 2.42, 1.57],  #-6
                               [15.12, 15.18, 12.30, 8.45, 3.64, 1.78, 1.06],  #-12
                               [5.03, 5.10, 4.42, 3.40, 2.11, 1.04, 1.04]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_capret_r1'] = [[42.02, 32.80, 18.22, 6.32, 1.44, 0.46, 0.26],  #18
                               [40.46, 32.48, 18.82, 6.12, 1.54, 0.32, 0.14],  #12
                               [38.98, 30.88, 17.38, 6.20, 1.58, 0.46, 0.20],  #6
                               [31.62, 25.24, 14.90, 4.82, 1.44, 0.30, 0.10],  #0
                               [15.56, 11.96, 6.92, 2.96, 1.20, 0.36, 0.24],  #-6
                               [2.44, 2.76, 1.78, 1.00, 0.40, 0.26, 0.18],  #-12
                               [0.48, 0.58, 0.62, 0.34, 0.16, 0.06, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_capret_r5'] = [[73.02, 59.98, 38.14, 13.90, 3.50, 0.90, 0.52],  #18
                               [71.00, 59.82, 38.34, 12.82, 3.36, 0.76, 0.42],  #12
                               [70.28, 58.12, 35.84, 13.06, 3.32, 1.04, 0.54],  #6
                               [60.24, 51.26, 32.44, 11.16, 3.36, 0.70, 0.42],  #0
                               [35.22, 28.90, 16.92, 6.54, 2.50, 0.60, 0.54],  #-6
                               [7.74, 7.72, 5.34, 3.48, 1.16, 0.52, 0.32],  #-12
                               [2.04, 1.90, 1.68, 1.16, 0.66, 0.22, 0.24]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb128_16_capret_r10'] = [[83.34, 73.44, 50.80, 20.96, 5.68, 1.52, 0.82],  #18
                               [82.92, 73.50, 50.90, 20.00, 5.72, 1.30, 0.68],  #12
                               [81.66, 72.02, 48.12, 19.04, 5.60, 1.58, 1.04],  #6
                               [73.58, 64.70, 44.66, 16.28, 5.12, 1.36, 0.74],  #0
                               [47.00, 39.78, 25.78, 10.26, 4.20, 1.08, 0.84],  #-6
                               [12.86, 12.30, 8.76, 5.82, 1.92, 0.96, 0.48],  #-12
                               [3.60, 3.22, 2.84, 1.98, 1.10, 0.40, 0.40]]   #-18
################################################################### 64 256 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_imgret_r1'] = [[62.60, 61.38, 59.69, 49.70, 26.38, 5.77, 1.11],  #18
                               [59.42, 58.84, 56.02, 47.05, 24.79, 6.26, 1.06],  #12
                               [48.21, 47.36, 45.59, 36.53, 19.40, 5.14, 1.03],  #6
                               [25.52, 25.08, 23.46, 19.65, 11.26, 3.10, 0.88],  #0
                               [7.00, 6.77, 6.66, 5.72, 3.77, 1.67, 0.36],  #-6
                               [1.32, 1.52, 1.65, 1.29, 1.25, 0.64, 0.28],  #-12
                               [0.35, 0.38, 0.42, 0.38, 0.33, 0.13, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_imgret_r5'] = [[87.26, 87.10, 85.73, 79.84, 57.08, 18.25, 3.74],  #18
                               [85.82, 85.44, 83.66, 77.22, 54.19, 20.11, 4.75],  #12
                               [79.46, 78.72, 76.98, 68.98, 46.64, 18.09, 4.62],  #6
                               [55.86, 55.47, 52.91, 46.59, 32.04, 11.68, 3.20],  #0
                               [21.74, 21.68, 20.70, 18.86, 13.80, 6.81, 1.97],  #-6
                               [5.62, 6.03, 5.80, 5.27, 4.90, 3.10, 1.12],  #-12
                               [1.67, 2.00, 1.92, 1.81, 1.66, 0.96, 1.00]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_imgret_r10'] = [[93.01, 93.01, 91.90, 87.81, 70.20, 27.71, 6.80],  #18
                               [92.08, 91.97, 90.58, 85.92, 67.01, 30.19, 8.62],  #12
                               [87.82, 87.20, 86.14, 80.31, 61.07, 27.76, 8.35],  #6
                               [69.45, 69.22, 66.61, 60.38, 45.90, 19.59, 6.01],  #0
                               [32.12, 32.71, 30.99, 29.00, 22.42, 12.08, 3.77],  #-6
                               [10.12, 10.50, 9.82, 9.28, 8.74, 5.61, 2.30],  #-12
                               [3.06, 3.78, 3.66, 3.42, 3.25, 2.11, 2.03]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_capret_r1'] = [[79.42, 76.20, 70.20, 51.78, 24.28, 5.28, 0.90],  #18
                               [76.32, 75.32, 67.68, 50.78, 22.42, 5.78, 0.94],  #12
                               [68.72, 65.70, 59.48, 42.14, 18.54, 4.80, 1.04],  #6
                               [39.92, 38.46, 33.78, 25.16, 12.42, 3.34, 0.76],  #0
                               [9.42, 9.90, 9.50, 6.88, 3.80, 1.62, 0.36],  #-6
                               [1.14, 1.60, 1.42, 1.04, 1.16, 0.76, 0.14],  #-12
                               [0.38, 0.44, 0.52, 0.26, 0.48, 0.20, 0.26]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_capret_r5'] = [[94.40, 93.46, 90.14, 77.12, 43.76, 10.98, 2.04],  #18
                               [94.12, 92.46, 88.94, 75.84, 41.48, 12.40, 2.48],  #12
                               [89.62, 88.18, 83.46, 67.88, 37.08, 10.30, 2.26],  #6
                               [70.22, 68.30, 62.54, 49.04, 27.32, 7.52, 1.96],  #0
                               [24.16, 24.70, 23.14, 17.72, 10.40, 4.46, 0.84],  #-6
                               [3.92, 5.52, 5.38, 3.76, 3.04, 2.32, 0.58],  #-12
                               [1.08, 1.62, 1.72, 1.02, 1.32, 0.58, 0.66]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_256_capret_r10'] = [[97.46, 97.18, 95.10, 87.26, 56.70, 16.38, 3.40],  #18
                               [97.30, 96.28, 94.44, 85.78, 53.60, 18.38, 4.32],  #12
                               [95.00, 94.08, 91.14, 79.36, 48.98, 15.84, 3.84],  #6
                               [80.10, 79.38, 74.54, 61.82, 38.14, 11.86, 3.46],  #0
                               [34.76, 35.66, 32.96, 26.86, 16.64, 7.24, 1.62],  #-6
                               [6.80, 8.76, 8.28, 6.44, 5.00, 3.66, 1.12],  #-12
                               [1.88, 2.72, 2.84, 1.82, 1.98, 0.90, 1.02]]   #-18
################################################################### 64 128 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_imgret_r1'] = [[61.82, 60.62, 55.27, 40.72, 16.06, 2.85, 0.50],  #18
                               [58.12, 57.19, 52.64, 37.13, 15.24, 2.86, 0.74],  #12
                               [47.58, 45.55, 41.48, 30.80, 11.91, 2.64, 0.57],  #6
                               [23.89, 23.28, 21.97, 15.73, 7.28, 1.89, 0.45],  #0
                               [6.92, 6.31, 5.68, 5.12, 2.77, 1.00, 0.30],  #-6
                               [1.22, 1.62, 1.48, 1.32, 0.94, 0.40, 0.20],  #-12
                               [0.51, 0.48, 0.42, 0.31, 0.28, 0.28, 0.11]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_imgret_r5'] = [[86.91, 86.50, 83.73, 72.71, 40.69, 10.42, 2.34],  #18
                               [85.26, 84.65, 81.52, 70.24, 39.37, 10.72, 3.13],  #12
                               [78.55, 77.12, 74.00, 61.54, 35.01, 10.42, 2.64],  #6
                               [54.75, 53.72, 51.67, 41.90, 23.27, 7.42, 2.35],  #0
                               [21.20, 20.58, 19.48, 17.19, 10.68, 4.33, 1.62],  #-6
                               [5.63, 6.24, 5.76, 5.10, 3.81, 2.20, 1.14],  #-12
                               [2.13, 2.25, 2.04, 1.65, 1.27, 1.04, 0.53]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_imgret_r10'] = [[92.81, 92.57, 90.58, 82.76, 53.86, 17.50, 4.32],  #18
                               [91.85, 91.40, 88.98, 80.60, 52.52, 17.68, 5.49],  #12
                               [87.30, 86.10, 84.19, 74.39, 48.57, 16.93, 4.77],  #6
                               [68.50, 67.38, 66.02, 56.02, 34.83, 12.94, 4.50],  #0
                               [32.03, 31.04, 29.51, 26.77, 17.38, 8.19, 3.06],  #-6
                               [9.52, 10.93, 9.70, 8.93, 6.51, 3.94, 2.17],  #-12
                               [3.88, 4.10, 3.70, 3.22, 2.55, 1.87, 1.01]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_capret_r1'] = [[76.68, 73.76, 63.38, 39.52, 13.80, 2.96, 0.56],  #18
                               [74.88, 71.76, 60.72, 37.04, 12.90, 2.54, 0.82],  #12
                               [65.94, 61.20, 53.02, 32.50, 10.46, 2.64, 0.46],  #6
                               [37.96, 35.62, 30.68, 18.84, 7.36, 1.92, 0.62],  #0
                               [9.70, 8.78, 7.52, 5.86, 3.02, 1.24, 0.22],  #-6
                               [1.44, 1.42, 1.36, 1.18, 0.72, 0.42, 0.26],  #-12
                               [0.50, 0.34, 0.38, 0.36, 0.16, 0.10, 0.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_capret_r5'] = [[94.22, 91.98, 85.84, 64.40, 26.70, 6.06, 0.90],  #18
                               [92.70, 91.14, 84.48, 61.90, 25.94, 5.50, 1.66],  #12
                               [89.04, 85.70, 77.76, 56.54, 22.70, 5.92, 1.38],  #6
                               [67.52, 63.82, 57.64, 38.74, 16.68, 4.46, 1.22],  #0
                               [24.32, 22.52, 20.20, 15.54, 7.58, 3.00, 0.66],  #-6
                               [4.32, 5.10, 4.64, 3.72, 2.42, 0.88, 0.62],  #-12
                               [1.32, 1.58, 1.36, 1.16, 0.50, 0.58, 0.16]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_128_capret_r10']  = [[97.50, 96.50, 93.02, 76.80, 37.24, 9.76, 1.60],  #18
                               [96.74, 95.78, 91.68, 74.28, 36.42, 8.78, 2.58],  #12
                               [94.26, 92.30, 87.32, 68.70, 32.46, 9.04, 2.30],  #6
                               [78.56, 75.80, 70.54, 51.98, 23.86, 7.00, 2.00],  #0
                               [34.88, 32.78, 29.50, 23.14, 11.80, 4.74, 1.22],  #-6
                               [7.06, 8.72, 7.46, 6.44, 4.08, 1.68, 1.04],  #-12
                               [2.26, 3.06, 2.68, 2.04, 1.16, 0.94, 0.32]]   #-18
################################################################### 64 64 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_imgret_r1'] = [[58.35, 57.12, 49.13, 28.42, 7.66, 1.23, 0.37],  #18
                               [55.17, 53.96, 45.56, 26.12, 8.20, 1.77, 0.39],  #12
                               [44.12, 42.43, 34.52, 21.64, 6.09, 1.43, 0.45],  #6
                               [24.72, 23.23, 17.82, 12.69, 4.46, 1.19, 0.19],  #0
                               [6.63, 6.16, 5.02, 4.22, 1.62, 0.74, 0.15],  #-6
                               [1.57, 1.38, 1.30, 1.02, 0.56, 0.18, 0.16],  #-12
                               [0.47, 0.40, 0.27, 0.40, 0.23, 0.10, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_imgret_r5'] = [[85.09, 84.46, 79.58, 59.37, 24.07, 5.51, 1.37],  #18
                               [83.62, 82.85, 76.49, 56.88, 24.41, 6.57, 1.80],  #12
                               [76.00, 75.18, 67.57, 50.39, 20.84, 5.29, 1.88],  #6
                               [54.34, 52.82, 45.36, 36.01, 15.39, 4.68, 1.39],  #0
                               [21.48, 20.70, 17.18, 14.96, 6.81, 3.16, 1.04],  #-6
                               [5.96, 5.46, 5.89, 4.26, 2.75, 1.11, 0.77],  #-12
                               [1.91, 2.03, 1.86, 1.59, 0.98, 0.63, 0.55]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_imgret_r10'] = [[91.44, 90.68, 87.54, 71.89, 35.04, 9.28, 2.69],  #18
                               [90.43, 89.98, 85.84, 70.13, 35.22, 11.28, 3.34],  #12
                               [85.53, 85.07, 79.19, 63.87, 31.55, 9.47, 3.60],  #6
                               [67.80, 66.43, 59.66, 50.68, 24.56, 8.62, 3.03],  #0
                               [32.78, 31.30, 27.27, 23.98, 12.27, 6.06, 1.84],  #-6
                               [10.20, 9.90, 9.93, 7.48, 5.22, 2.35, 2.06],  #-12
                               [3.52, 3.84, 3.61, 3.09, 2.04, 1.16, 0.98]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_capret_r1'] = [[71.12, 66.72, 50.96, 23.90, 6.28, 1.26, 0.38],  #18
                               [70.18, 64.88, 49.46, 22.48, 6.58, 1.76, 0.52],  #12
                               [61.00, 57.30, 40.72, 20.06, 5.96, 1.10, 0.46],  #6
                               [38.68, 35.74, 22.80, 14.48, 3.66, 1.18, 0.24],  #0
                               [10.02, 9.12, 6.38, 4.74, 1.88, 0.76, 0.20],  #-6
                               [1.48, 1.34, 1.68, 0.88, 0.70, 0.24, 0.24],  #-12
                               [0.30, 0.38, 0.28, 0.32, 0.24, 0.04, 0.12]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_capret_r5'] = [[92.84, 89.10, 77.68, 45.50, 13.32, 2.68, 0.68],  #18
                               [91.16, 88.02, 75.14, 42.28, 14.06, 3.28, 1.00],  #12
                               [86.02, 81.32, 67.14, 38.98, 12.14, 2.56, 1.14],  #6
                               [66.72, 62.18, 46.50, 30.78, 8.84, 2.72, 0.60],  #0
                               [24.92, 23.50, 17.24, 11.90, 4.94, 1.56, 0.48],  #-6
                               [4.76, 4.98, 5.14, 2.84, 2.04, 0.76, 0.38],  #-12
                               [1.18, 1.68, 1.04, 0.80, 0.56, 0.12, 0.28]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_64_capret_r10'] = [[96.62, 94.98, 86.88, 58.22, 20.04, 4.72, 1.24],  #18
                               [95.62, 93.56, 85.34, 55.70, 20.78, 5.36, 1.80],  #12
                               [93.00, 89.80, 78.04, 51.84, 18.98, 4.78, 1.60],  #6
                               [78.62, 73.52, 60.26, 41.96, 14.44, 4.56, 1.02],  #0
                               [35.00, 32.86, 26.00, 18.24, 7.62, 3.00, 0.92],  #-6
                               [7.98, 8.28, 8.14, 4.44, 3.12, 1.26, 0.68],  #-12
                               [2.16, 2.84, 2.16, 1.56, 1.02, 0.28, 0.46]]   #-18
# di, dc = 64, 64
# SNR_X, SNR_Y = np.meshgrid(snr, snr)
# fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
# for j, ret in enumerate(['img', 'cap']):
#     for i, r in enumerate([1, 5, 10]):
#         ax[j, i].plot_surface(SNR_X, SNR_Y, np.array(acc[f'symb{dc}_{dc}_{ret}ret_r{r}']), cmap=cm.coolwarm)
#         ax[j, i].set_xlabel('cap SNR')
#         ax[j, i].set_ylabel('img SNR')
#         ax[j, i].set_zlabel('acc/%')
#         ax[j, i].set_title(f'{ret} retrieval R@{r}')
#         ax[j, i].legend(loc='upper left', fontsize='small')
#         ax[j, i].grid()
# fig.suptitle('acc when img and cap transmitted in different SNR')
# plt.show()
################################################################### 64 32 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_imgret_r1'] = [[51.82, 49.68, 37.78, 16.33, 3.65, 0.75, 0.23],  #18
                               [48.88, 45.66, 35.12, 15.50, 3.11, 0.68, 0.22],  #12
                               [39.95, 36.96, 28.66, 12.86, 2.49, 0.69, 0.28],  #6
                               [21.12, 19.50, 15.22, 7.50, 1.83, 0.60, 0.21],  #0
                               [5.92, 5.86, 4.71, 2.67, 1.08, 0.34, 0.30],  #-6
                               [1.30, 1.34, 1.05, 0.72, 0.48, 0.16, 0.10],  #-12
                               [0.39, 0.55, 0.43, 0.32, 0.22, 0.12, 0.11]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_imgret_r5'] = [[81.89, 80.03, 70.08, 40.77, 12.96, 3.01, 1.41],  #18
                               [80.25, 77.72, 67.96, 40.76, 12.36, 2.95, 1.17],  #12
                               [72.70, 70.70, 59.99, 35.42, 10.55, 2.95, 1.22],  #6
                               [50.32, 47.46, 40.04, 23.62, 7.64, 2.33, 0.91],  #0
                               [19.26, 19.25, 16.27, 10.01, 4.40, 1.78, 1.15],  #-6
                               [5.74, 5.50, 5.15, 3.74, 2.11, 0.91, 0.46],  #-12
                               [1.92, 2.14, 1.78, 1.43, 1.00, 0.53, 0.48]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_imgret_r10'] = [[89.36, 87.86, 80.74, 54.74, 20.94, 5.84, 2.64],  #18
                               [88.43, 86.42, 79.00, 54.66, 20.08, 5.53, 2.54],  #12
                               [83.24, 81.66, 73.01, 49.16, 17.77, 5.55, 2.50],  #6
                               [64.42, 61.42, 54.04, 35.06, 13.05, 4.48, 1.96],  #0
                               [29.62, 29.31, 25.18, 16.74, 7.70, 3.05, 2.07],  #-6
                               [9.85, 9.54, 9.33, 7.10, 3.87, 1.89, 0.87],  #-12
                               [3.74, 3.82, 3.33, 2.87, 2.23, 1.07, 1.04]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_capret_r1'] = [[64.12, 56.90, 35.98, 13.08, 3.34, 0.78, 0.34],  #18
                               [62.44, 53.22, 34.08, 12.68, 3.02, 0.70, 0.26],  #12
                               [55.00, 47.54, 29.90, 11.12, 3.02, 0.72, 0.24],  #6
                               [32.48, 26.84, 17.54, 7.22, 2.02, 0.62, 0.18],  #0
                               [8.04, 7.88, 5.20, 2.86, 0.98, 0.38, 0.32],  #-6
                               [1.28, 0.94, 1.12, 0.88, 0.48, 0.16, 0.10],  #-12
                               [0.38, 0.42, 0.34, 0.24, 0.24, 0.16, 0.04]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_capret_r5'] = [[88.86, 81.78, 61.14, 26.06, 7.06, 1.92, 0.78],  #18
                               [87.52, 79.60, 59.66, 26.94, 6.48, 1.46, 0.46],  #12
                               [82.62, 75.06, 54.22, 23.72, 6.22, 1.40, 0.48],  #6
                               [59.76, 53.22, 37.58, 16.12, 4.40, 1.36, 0.34],  #0
                               [21.54, 19.92, 14.08, 7.38, 2.60, 0.76, 0.56],  #-6
                               [4.56, 4.54, 3.86, 2.72, 1.40, 0.40, 0.22],  #-12
                               [1.68, 1.42, 1.18, 0.94, 0.56, 0.28, 0.20]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_32_capret_r10'] = [[94.76, 90.66, 73.62, 36.18, 10.82, 3.02, 1.30],  #18
                               [93.26, 88.54, 72.46, 37.56, 9.98, 2.62, 0.98],  #12
                               [90.36, 84.80, 66.86, 33.50, 9.64, 2.48, 0.90],  #6
                               [73.26, 66.04, 50.20, 24.44, 6.84, 2.20, 0.86],  #0
                               [31.14, 29.48, 22.20, 11.66, 4.76, 1.48, 0.92],  #-6
                               [7.54, 7.86, 6.86, 4.60, 2.26, 0.86, 0.44],  #-12
                               [2.90, 2.58, 2.02, 1.56, 0.98, 0.46, 0.38]]   #-18
################################################################### 64 16 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_imgret_r1'] = [[35.64, 32.66, 22.38, 7.40, 1.42, 0.49, 0.28],  #18
                               [33.85, 30.32, 20.46, 7.14, 1.49, 0.45, 0.25],  #12
                               [27.52, 25.02, 17.57, 5.86, 0.98, 0.40, 0.20],  #6
                               [17.18, 14.90, 11.30, 3.64, 1.15, 0.34, 0.15],  #0
                               [5.78, 4.95, 3.72, 1.72, 0.65, 0.23, 0.24],  #-6
                               [1.54, 1.40, 1.30, 0.62, 0.19, 0.23, 0.12],  #-12
                               [0.60, 0.28, 0.30, 0.19, 0.19, 0.13, 0.10]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_imgret_r5'] = [[69.80, 64.89, 52.30, 24.56, 6.31, 2.13, 1.04],  #18
                               [67.59, 62.65, 49.74, 22.51, 6.04, 1.90, 1.09],  #12
                               [59.85, 57.20, 45.11, 19.69, 5.13, 2.11, 1.00],  #6
                               [43.26, 40.34, 32.62, 14.03, 4.73, 1.59, 0.86],  #0
                               [18.54, 17.06, 13.27, 6.84, 3.20, 0.95, 0.91],  #-6
                               [5.61, 5.04, 4.66, 2.74, 1.07, 1.05, 0.60],  #-12
                               [2.02, 1.68, 1.63, 1.18, 1.04, 0.48, 0.39]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_imgret_r10'] = [[81.18, 77.26, 66.04, 36.48, 10.69, 3.86, 2.15],  #18
                               [79.75, 75.58, 64.42, 33.90, 10.34, 3.84, 2.15],  #12
                               [73.40, 70.85, 59.67, 30.01, 9.47, 3.77, 1.96],  #6
                               [58.00, 55.08, 46.13, 22.96, 8.22, 3.32, 1.71],  #0
                               [28.80, 26.73, 21.60, 12.30, 5.92, 2.02, 1.72],  #-6
                               [9.60, 8.94, 8.23, 5.30, 2.00, 2.03, 1.18],  #-12
                               [3.82, 3.26, 3.42, 2.13, 1.98, 0.97, 0.82]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_capret_r1'] = [[41.22, 32.30, 18.28, 6.36, 1.58, 0.54, 0.26],  #18
                               [40.56, 31.46, 17.44, 5.88, 1.34, 0.44, 0.30],  #12
                               [36.20, 28.48, 16.24, 5.16, 1.26, 0.56, 0.24],  #6
                               [25.08, 20.18, 12.50, 3.36, 0.96, 0.44, 0.18],  #0
                               [8.50, 6.54, 4.46, 1.68, 0.80, 0.24, 0.18],  #-6
                               [1.32, 1.00, 1.14, 0.74, 0.26, 0.16, 0.02],  #-12
                               [0.28, 0.40, 0.32, 0.16, 0.26, 0.10, 0.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_capret_r5'] = [[72.12, 60.56, 37.94, 13.58, 3.36, 1.12, 0.48],  #18
                               [70.92, 59.80, 36.32, 12.86, 2.78, 0.80, 0.50],  #12
                               [65.78, 56.22, 34.66, 11.54, 3.02, 1.20, 0.60],  #6
                               [52.16, 43.34, 27.30, 8.80, 2.56, 0.88, 0.48],  #0
                               [20.04, 16.70, 11.28, 4.60, 1.86, 0.54, 0.42],  #-6
                               [4.88, 3.86, 3.24, 2.00, 0.70, 0.56, 0.16],  #-12
                               [1.38, 1.20, 1.32, 0.54, 0.62, 0.24, 0.34]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb64_16_capret_r10'] = [[83.28, 73.36, 50.58, 20.84, 5.62, 1.94, 0.90],  #18
                               [81.74, 72.02, 48.44, 19.56, 5.06, 1.50, 1.02],  #12
                               [77.74, 69.18, 46.02, 17.38, 4.64, 1.68, 0.96],  #6
                               [64.92, 55.42, 38.58, 13.92, 4.30, 1.42, 0.84],  #0
                               [29.64, 25.00, 17.22, 7.34, 3.26, 0.86, 0.72],  #-6
                               [7.88, 6.42, 5.48, 3.36, 1.18, 1.06, 0.56],  #-12
                               [2.46, 2.50, 2.10, 1.20, 1.00, 0.42, 0.50]]   #-18
################################################################### 32 256 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_imgret_r1'] = [[53.23, 53.20, 51.46, 44.01, 25.56, 5.40, 1.02],  #18
                               [47.48, 47.27, 46.54, 39.43, 21.52, 5.03, 1.14],  #12
                               [34.25, 33.29, 32.53, 28.08, 15.19, 4.09, 0.75],  #6
                               [13.44, 13.25, 13.02, 11.22, 6.91, 2.34, 0.57],  #0
                               [3.43, 3.42, 2.84, 2.70, 1.94, 0.88, 0.26],  #-6
                               [0.69, 0.82, 0.82, 0.87, 0.60, 0.32, 0.06],  #-12
                               [0.34, 0.32, 0.34, 0.31, 0.18, 0.25, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_imgret_r5'] = [[83.32, 82.70, 81.06, 75.61, 55.92, 18.71, 3.83],  #18
                               [79.62, 78.69, 77.60, 71.60, 50.72, 17.72, 4.72],  #12
                               [66.94, 65.68, 65.16, 58.69, 40.34, 14.62, 3.46],  #6
                               [36.08, 35.96, 35.21, 31.44, 21.66, 8.65, 2.34],  #0
                               [12.08, 11.41, 10.97, 9.79, 7.77, 4.61, 1.24],  #-6
                               [3.46, 3.35, 3.32, 3.57, 2.60, 1.65, 0.54],  #-12
                               [1.53, 1.61, 1.50, 1.40, 0.94, 1.17, 0.61]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_imgret_r10'] = [[90.46, 89.94, 88.87, 84.95, 68.88, 28.52, 6.82],  #18
                               [88.02, 87.68, 86.62, 82.12, 65.14, 27.98, 8.44],  #12
                               [79.18, 77.92, 77.65, 71.78, 54.27, 23.81, 6.52],  #6
                               [50.31, 50.19, 48.72, 44.84, 32.70, 14.71, 4.62],  #0
                               [19.19, 18.35, 17.68, 15.94, 13.54, 8.52, 2.18],  #-6
                               [6.09, 6.01, 6.16, 6.14, 5.19, 3.30, 1.06],  #-12
                               [2.94, 2.78, 2.72, 2.76, 2.07, 1.95, 1.37]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_capret_r1'] = [[68.76, 67.06, 61.54, 44.78, 22.16, 4.94, 1.16],  #18
                               [65.74, 61.66, 57.86, 42.96, 20.58, 5.26, 1.06],  #12
                               [51.24, 49.90, 46.56, 33.50, 14.82, 3.70, 0.80],  #6
                               [21.78, 20.70, 19.54, 14.74, 7.28, 2.12, 0.62],  #0
                               [4.10, 3.60, 3.60, 2.74, 2.18, 1.20, 0.20],  #-6
                               [0.82, 0.78, 0.56, 0.50, 0.56, 0.38, 0.18],  #-12
                               [0.32, 0.26, 0.30, 0.18, 0.18, 0.12, 0.16]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_capret_r5'] = [[91.62, 90.24, 86.38, 72.68, 42.52, 10.80, 2.22],  #18
                               [88.82, 86.84, 83.12, 70.40, 38.54, 11.06, 2.50],  #12
                               [80.02, 78.04, 74.08, 59.40, 32.18, 8.72, 1.86],  #6
                               [47.20, 45.30, 41.80, 33.20, 17.44, 5.76, 1.48],  #0
                               [12.72, 11.56, 10.94, 8.60, 5.88, 3.44, 0.44],  #-6
                               [2.62, 2.76, 2.56, 1.72, 1.74, 0.88, 0.20],  #-12
                               [1.00, 1.00, 1.18, 0.84, 0.60, 0.34, 0.22]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_256_capret_r10'] = [[96.28, 95.18, 93.00, 82.68, 55.26, 16.36, 3.76],  #18
                               [94.30, 93.84, 90.74, 81.56, 50.96, 16.42, 4.26],  #12
                               [88.36, 87.18, 84.04, 72.44, 44.26, 13.44, 3.34],  #6
                               [60.02, 58.98, 54.80, 45.16, 26.02, 9.46, 2.22],  #0
                               [19.16, 17.66, 16.70, 13.72, 9.70, 5.24, 1.04],  #-6
                               [4.30, 4.94, 4.44, 3.36, 2.94, 1.70, 0.48],  #-12
                               [1.90, 1.86, 2.12, 1.64, 1.08, 0.68, 0.42]]   #-18
################################################################### 32 128 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_imgret_r1'] = [[53.37, 51.76, 48.48, 36.97, 14.73, 2.42, 0.58],  #18
                               [48.34, 45.97, 43.16, 32.48, 12.84, 3.02, 0.56],  #12
                               [32.58, 32.70, 30.03, 22.07, 9.62, 2.00, 0.54],  #6
                               [13.12, 12.93, 12.16, 9.22, 4.73, 1.02, 0.50],  #0
                               [3.03, 3.12, 2.90, 2.30, 1.68, 0.83, 0.14],  #-6
                               [0.75, 0.76, 0.83, 0.73, 0.61, 0.36, 0.20],  #-12
                               [0.25, 0.31, 0.25, 0.22, 0.26, 0.24, 0.13]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_imgret_r5'] = [[82.25, 81.64, 78.81, 69.67, 38.84, 9.44, 2.63],  #18
                               [79.61, 77.88, 75.57, 64.94, 36.46, 11.21, 2.78],  #12
                               [66.06, 64.91, 61.61, 51.72, 29.20, 9.11, 2.59],  #6
                               [35.80, 35.97, 33.47, 27.35, 16.39, 5.14, 1.93],  #0
                               [10.77, 11.31, 10.64, 9.04, 6.70, 3.36, 1.14],  #-6
                               [3.53, 3.28, 3.36, 3.26, 2.54, 1.59, 1.06],  #-12
                               [1.27, 1.32, 1.33, 1.23, 1.22, 0.86, 0.56]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_imgret_r10'] = [[89.92, 89.43, 87.29, 80.26, 52.52, 16.18, 4.65],  #18
                               [87.93, 87.00, 84.98, 76.94, 50.16, 18.29, 5.16],  #12
                               [78.26, 77.09, 74.24, 65.77, 42.16, 15.66, 4.98],  #6
                               [49.24, 50.04, 46.96, 40.40, 26.36, 9.30, 3.68],  #0
                               [17.78, 18.40, 17.16, 15.37, 11.67, 6.50, 2.20],  #-6
                               [6.40, 5.98, 6.15, 5.80, 4.70, 3.05, 2.04],  #-12
                               [2.56, 2.61, 2.29, 2.34, 1.96, 1.76, 0.96]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_capret_r1'] = [[67.48, 64.40, 54.26, 35.64, 12.32, 2.54, 0.78],  #18
                               [65.00, 60.12, 50.54, 32.10, 11.62, 2.74, 0.84],  #12
                               [49.76, 47.20, 39.32, 24.76, 9.14, 1.94, 0.52],  #6
                               [20.86, 20.56, 17.68, 10.32, 4.90, 1.32, 0.52],  #0
                               [4.12, 3.58, 2.84, 2.34, 1.60, 0.84, 0.22],  #-6
                               [0.80, 0.72, 0.62, 0.70, 0.42, 0.44, 0.16],  #-12
                               [0.36, 0.26, 0.22, 0.22, 0.08, 0.16, 0.10]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_capret_r5'] = [[90.50, 88.04, 81.10, 61.82, 25.48, 5.62, 1.46],  #18
                               [88.32, 85.54, 78.22, 57.66, 23.68, 5.98, 1.48],  #12
                               [78.16, 74.66, 67.68, 46.62, 19.78, 4.68, 1.34],  #6
                               [45.02, 43.98, 37.92, 25.56, 11.46, 2.98, 1.06],  #0
                               [11.44, 11.58, 9.28, 7.02, 4.94, 2.28, 0.44],  #-6
                               [2.64, 2.76, 2.14, 2.12, 1.64, 1.08, 0.46],  #-12
                               [1.10, 1.24, 0.92, 0.76, 0.62, 0.42, 0.20]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_128_capret_r10'] = [[94.76, 93.94, 89.60, 73.22, 34.64, 8.86, 2.12],  #18
                               [94.04, 92.48, 87.48, 70.68, 33.26, 9.28, 2.56],  #12
                               [87.48, 84.92, 78.66, 60.68, 28.80, 7.94, 2.30],  #6
                               [58.16, 57.36, 49.66, 36.56, 17.92, 4.86, 1.84],  #0
                               [17.62, 17.62, 15.50, 11.06, 7.84, 3.88, 1.04],  #-6
                               [4.74, 4.94, 4.12, 3.44, 2.84, 1.66, 0.86],  #-12
                               [1.90, 2.30, 1.42, 1.42, 1.10, 0.66, 0.28]]   #-18
################################################################### 32 64 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_imgret_r1'] = [[52.81, 50.37, 44.04, 26.50, 7.71, 1.56, 0.34],  #18
                               [46.16, 44.87, 38.05, 23.18, 7.18, 1.26, 0.37],  #12
                               [32.07, 30.50, 26.88, 15.76, 4.91, 1.01, 0.45],  #6
                               [12.96, 12.32, 10.63, 7.09, 2.59, 0.94, 0.29],  #0
                               [3.06, 3.08, 2.75, 1.89, 0.83, 0.15, 0.13],  #-6
                               [0.71, 0.91, 0.66, 0.55, 0.33, 0.12, 0.07],  #-12
                               [0.27, 0.29, 0.22, 0.22, 0.23, 0.11, 0.12]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_imgret_r5'] = [[81.96, 80.57, 75.72, 57.34, 23.83, 6.42, 1.60],  #18
                               [77.80, 76.69, 70.63, 53.13, 22.78, 5.37, 1.93],  #12
                               [64.73, 63.04, 58.71, 40.91, 17.03, 4.79, 1.74],  #6
                               [35.39, 34.02, 31.12, 22.73, 10.33, 3.71, 1.39],  #0
                               [11.37, 11.04, 9.89, 7.86, 4.20, 1.22, 1.07],  #-6
                               [3.31, 3.62, 2.82, 2.45, 1.60, 0.96, 0.46],  #-12
                               [1.45, 1.36, 1.13, 1.03, 1.06, 0.44, 0.50]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_imgret_r10'] = [[89.46, 88.43, 85.12, 70.26, 34.91, 11.12, 3.17],  #18
                               [87.09, 86.25, 81.56, 66.54, 34.31, 9.91, 3.84],  #12
                               [77.38, 76.08, 71.73, 55.05, 26.72, 8.54, 3.25],  #6
                               [48.90, 47.44, 44.10, 34.32, 17.62, 6.82, 2.55],  #0
                               [18.31, 18.07, 16.42, 13.24, 7.94, 2.56, 2.09],  #-6
                               [5.94, 6.29, 5.72, 4.39, 3.14, 1.91, 0.86],  #-12
                               [2.85, 2.83, 2.20, 2.01, 2.02, 0.98, 0.92]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_capret_r1'] = [[66.04, 60.10, 46.64, 23.18, 5.96, 1.40, 0.58],  #18
                               [60.88, 56.66, 42.76, 21.22, 6.28, 1.42, 0.48],  #12
                               [48.18, 43.96, 33.72, 15.38, 4.68, 1.20, 0.50],  #6
                               [19.98, 19.00, 13.74, 7.98, 2.86, 0.80, 0.28],  #0
                               [3.64, 3.32, 2.52, 2.28, 0.80, 0.26, 0.24],  #-6
                               [0.76, 0.60, 0.66, 0.52, 0.32, 0.24, 0.10],  #-12
                               [0.24, 0.40, 0.22, 0.12, 0.20, 0.04, 0.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_capret_r5'] = [[89.24, 86.12, 73.06, 44.56, 13.38, 3.04, 1.12],  #18
                               [85.86, 83.34, 69.06, 40.60, 13.24, 3.00, 0.82],  #12
                               [76.52, 72.22, 59.34, 32.28, 10.72, 2.52, 1.04],  #6
                               [43.44, 40.26, 31.92, 18.66, 6.86, 1.90, 0.64],  #0
                               [11.74, 11.00, 8.28, 5.86, 2.58, 0.64, 0.62],  #-6
                               [2.52, 2.90, 2.36, 1.68, 0.92, 0.56, 0.24],  #-12
                               [1.04, 1.26, 0.94, 0.68, 0.44, 0.16, 0.34]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_64_capret_r10'] = [[94.64, 92.78, 83.76, 56.42, 20.36, 5.30, 1.90],  #18
                               [92.46, 90.88, 80.44, 52.94, 20.24, 5.00, 1.50],  #12
                               [86.22, 83.02, 71.84, 44.06, 16.24, 4.40, 1.62],  #6
                               [56.22, 53.02, 44.18, 27.62, 10.76, 3.18, 1.18],  #0
                               [18.08, 17.12, 13.14, 9.76, 4.74, 1.14, 1.06],  #-6
                               [4.30, 5.04, 3.82, 2.64, 1.60, 1.08, 0.42],  #-12
                               [1.84, 2.10, 1.70, 1.28, 0.82, 0.32, 0.50]]   #-18
################################################################### 32 32 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_imgret_r1'] = [[48.98, 44.31, 35.07, 15.14, 3.34,  0.70, 0.29],  #18
                               [44.19, 39.25, 30.43, 13.38, 2.91, 0.64, 0.39],  #12
                               [29.09, 27.96, 22.94, 13.07, 2.27, 0.52, 0.31],  #6
                               [12.30, 10.04, 8.78, 4.68, 1.38, 0.58, 0.19],  #0
                               [2.82, 3.23, 2.20, 1.56, 0.80, 0.20, 0.27],  #-6
                               [0.71, 0.83, 0.60, 0.37, 0.38, 0.35, 0.38],  #-12
                               [0.40, 0.32, 0.33, 0.28, 0.44, 0.26, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_imgret_r5'] = [[79.54, 76.75, 67.35, 39.80, 12.26, 3.28, 1.23],  #18
                               [76.42, 72.39, 62.85, 37.24, 11.68, 3.09, 1.59],  #12
                               [61.37, 59.88, 52.92, 34.62, 9.91, 2.84, 1.19],  #6
                               [33.32, 29.96, 27.06, 16.68, 5.95, 2.34, 1.26],  #0
                               [10.77, 11.16, 8.36, 5.83, 3.45, 1.06, 1.44],  #-6
                               [3.70, 3.47, 2.82, 2.04, 1.45, 1.22, 1.81],  #-12
                               [1.41, 1.40, 1.39, 1.22, 1.94, 0.82, 0.68]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_imgret_r10'] = [[88.00, 85.82, 78.36, 53.26, 19.72, 5.77, 2.34],  #18
                               [86.07, 83.08, 75.41, 50.61, 19.16, 5.77, 2.87],  #12
                               [74.78, 73.28, 66.89, 47.85, 16.84, 5.36, 2.44],  #6
                               [47.12, 43.25, 39.58, 26.48, 10.67, 4.10, 2.54],  #0
                               [17.77, 18.40, 14.23, 10.16, 6.46, 2.14, 2.41],  #-6
                               [6.78, 6.15, 5.17, 3.94, 2.63, 2.30, 3.27],  #-12
                               [2.64, 2.70, 2.48, 2.44, 3.50, 1.45, 1.28]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_capret_r1'] = [[60.80, 50.26, 33.36, 12.12, 2.90, 0.60, 0.34],  #18
                               [57.96, 46.54, 29.44, 11.04, 2.84, 0.66, 0.24],  #12
                               [43.50, 36.86, 25.44, 11.34, 2.66, 0.66, 0.18],  #6
                               [19.76, 14.48, 10.42, 5.08, 1.60, 0.58, 0.26],  #0
                               [3.74, 3.64, 2.32, 1.40, 0.78, 0.16, 0.28],  #-6
                               [0.66, 0.86, 0.56, 0.32, 0.30, 0.38, 0.28],  #-12
                               [0.24, 0.24, 0.34, 0.34, 0.34, 0.16, 0.18]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_capret_r5'] = [[86.36, 78.00, 58.12, 25.28, 6.48, 1.28, 0.76],  #18
                               [84.60, 74.72, 54.28, 23.34, 6.46, 1.66, 0.64],  #12
                               [72.86, 64.48, 49.70, 24.14, 5.48, 1.30, 0.48],  #6
                               [42.54, 33.68, 25.06, 12.02, 3.62, 1.36, 0.64],  #0
                               [11.46, 10.70, 7.42, 3.78, 1.92, 0.58, 0.60],  #-6
                               [2.98, 2.84, 1.92, 1.16, 0.70, 0.60, 0.52],  #-12
                               [0.96, 0.94, 0.98, 1.14, 0.96, 0.34, 0.36]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_32_capret_r10'] = [[92.90, 87.52, 70.16, 36.04, 10.34, 2.64, 1.34],  #18
                               [91.72, 84.74, 68.32, 33.28, 10.22, 2.74, 1.18],  #12
                               [83.26, 77.18, 62.62, 33.94, 8.44, 2.18, 0.96],  #6
                               [55.66, 45.96, 35.22, 18.58, 5.98, 2.10, 1.24],  #0
                               [17.70, 16.60, 11.62, 6.98, 3.32, 0.98, 1.12],  #-6
                               [5.12, 4.76, 3.22, 2.04, 1.10, 0.82, 1.02],  #-12
                               [1.98, 1.84, 1.76, 1.92, 1.66, 0.58, 0.60]]   #-18
################################################################### 32 16 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_imgret_r1'] = [[34.60, 30.96, 20.73, 7.07, 1.39, 0.43, 0.30],  #18
                               [30.75, 27.38, 18.18, 6.71, 1.56, 0.45, 0.08],  #12
                               [20.94, 19.68, 13.56, 5.06, 1.05, 0.31, 0.18],  #6
                               [10.20, 9.00, 6.07, 2.75, 0.90, 0.31, 0.22],  #0
                               [2.58, 2.52, 1.83, 1.00, 0.51, 0.23, 0.08],  #-6
                               [0.77, 0.57, 0.61, 0.36, 0.16, 0.12, 0.09],  #-12
                               [0.30, 0.34, 0.16, 0.20, 0.12, 0.16, 0.12]]   #-18

###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_imgret_r5'] = [[68.74, 64.26, 50.96, 22.78, 5.86, 1.90, 0.99],  #18
                               [64.54, 59.98, 46.98, 21.16, 6.04, 1.78, 0.92],  #12
                               [51.24, 48.92, 37.09, 17.18, 5.12, 1.92, 0.84],  #6
                               [29.99, 28.22, 20.75, 9.77, 4.02, 1.34, 0.79],  #0
                               [9.89, 8.99, 7.73, 4.46, 2.13, 0.93, 0.38],  #-6
                               [3.33, 2.86, 2.70, 1.72, 0.96, 0.90, 0.52],  #-12
                               [1.42, 1.21, 1.08, 1.01, 0.54, 0.59, 0.58]]   #-18

###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_imgret_r10'] = [[80.78, 77.17, 64.96, 34.61, 10.48, 3.64, 1.87],  #18
                               [77.34, 73.64, 60.91, 32.42, 10.46, 3.59, 1.89],  #12
                               [65.99, 63.66, 51.93, 27.16, 9.16, 3.67, 1.62],  #6
                               [43.16, 41.06, 31.74, 16.66, 7.30, 2.73, 1.54],  #0
                               [16.67, 15.77, 12.96, 8.16, 4.11, 1.92, 0.82],  #-6
                               [5.95, 5.22, 4.84, 3.33, 1.85, 1.73, 1.01],  #-12
                               [2.77, 2.41, 1.98, 2.09, 1.04, 1.11, 1.13]]   #-18

###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_capret_r1'] = [[41.42, 32.14, 18.78, 6.10, 1.60, 0.40, 0.18],  #18
                               [39.02, 30.16, 16.64, 5.68, 1.42, 0.54, 0.20],  #12
                               [30.44, 24.70, 13.82, 4.70, 1.26, 0.34, 0.16],  #6
                               [15.42, 12.64, 6.94, 2.62, 1.24, 0.20, 0.08],  #0
                               [2.86, 2.74, 2.14, 0.86, 0.44, 0.12, 0.16],  #-6
                               [0.82, 0.54, 0.48, 0.30, 0.24, 0.08, 0.12],  #-12
                               [0.26, 0.12, 0.22, 0.24, 0.16, 0.18, 0.08]]   #-18

###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_capret_r5'] = [[71.76, 59.86, 36.88, 12.88, 3.38, 0.96, 0.64],  #18
                               [68.62, 57.26, 34.64, 12.40, 2.96, 1.02, 0.68],  #12
                               [60.68, 50.14, 30.04, 9.96, 2.86, 0.80, 0.40],  #6
                               [36.70, 29.74, 16.94, 6.76, 2.36, 0.50, 0.34],  #0
                               [9.12, 8.38, 6.36, 2.60, 1.30, 0.40, 0.38],  #-6
                               [2.88, 2.20, 1.70, 1.06, 0.72, 0.34, 0.48],  #-12
                               [1.04, 0.92, 0.68, 0.74, 0.42, 0.44, 0.46]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb32_16_capret_r10'] = [[83.02, 72.80, 48.84, 19.66, 5.30, 1.56, 1.16],  #18
                               [80.52, 70.88, 46.34, 18.22, 4.74, 1.62, 1.08],  #12
                               [73.52, 63.92, 41.68, 15.60, 4.70, 1.24, 0.64],  #6
                               [48.74, 41.34, 24.40, 10.68, 3.64, 1.02, 0.78],  #0
                               [14.46, 13.22, 9.70, 4.94, 1.94, 0.76, 0.72],  #-6
                               [4.80, 3.84, 3.12, 1.70, 1.12, 0.78, 0.72],  #-12
                               [1.90, 1.84, 1.52, 1.32, 0.62, 0.70, 0.92]]   #-18
################################################################### 16 256 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_imgret_r1'] = [[35.02, 34.22, 33.94, 29.68, 19.14, 4.89, 0.78],  #18
                               [29.08, 29.35, 28.45, 24.48, 15.72, 4.84, 0.70],  #12
                               [17.32, 17.57, 17.62, 15.23, 10.24, 2.98, 0.65],  #6
                               [6.18, 6.08, 5.86, 5.17, 3.62, 1.30, 0.42],  #0
                               [1.38, 1.36, 1.41, 1.26, 1.02, 0.62, 0.23],  #-6
                               [0.58, 0.40, 0.48, 0.47, 0.44, 0.10, 0.08],  #-12
                               [0.28, 0.23, 0.16, 0.26, 0.17, 0.22, 0.09]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_imgret_r5'] = [[69.38, 68.34, 67.32, 63.16, 48.15, 16.66, 3.38],  #18
                               [62.92, 62.26, 61.75, 56.24, 41.90, 16.79, 3.38],  #12
                               [44.75, 44.62, 44.52, 40.54, 31.01, 11.12, 3.03],  #6
                               [20.67, 20.18, 18.76, 18.00, 13.28, 5.74, 1.85],  #0
                               [6.11, 5.82, 5.66, 5.24, 4.27, 2.58, 0.91],  #-6
                               [2.49, 2.18, 2.26, 1.93, 1.84, 0.94, 0.39],  #-12
                               [1.35, 1.11, 0.97, 1.12, 0.89, 0.91, 0.50]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_imgret_r10'] = [[80.71, 80.20, 79.48, 75.98, 62.48, 25.74, 6.20],  #18
                               [75.71, 75.18, 75.20, 69.93, 56.47, 26.12, 6.32],  #12
                               [59.56, 59.16, 59.43, 55.12, 43.98, 18.98, 5.61],  #6
                               [31.34, 30.82, 29.15, 28.13, 21.48, 10.27, 3.52],  #0
                               [10.62, 10.42, 9.75, 9.58, 7.80, 5.16, 2.14],  #-6
                               [4.46, 4.00, 4.09, 3.41, 3.70, 1.90, 0.82],  #-12
                               [2.36, 2.02, 1.89, 2.04, 1.80, 1.89, 1.00]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_capret_r1'] = [[48.16, 44.50, 40.24, 30.46, 17.32, 4.74, 0.98],  #18
                               [42.96, 42.04, 37.12, 27.44, 15.20, 4.70, 0.72],  #12
                               [29.70, 28.04, 26.16, 19.66, 11.04, 3.00, 0.82],  #6
                               [9.68, 9.60, 8.04, 7.22, 3.72, 1.32, 0.42],  #0
                               [1.48, 1.52, 1.58, 1.40, 1.12, 0.58, 0.14],  #-6
                               [0.44, 0.38, 0.42, 0.28, 0.38, 0.20, 0.16],  #-12
                               [0.24, 0.12, 0.14, 0.22, 0.20, 0.12, 0.12]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_capret_r5'] = [[79.12, 75.78, 70.86, 58.18, 35.06, 10.38, 2.16],  #18
                               [75.30, 72.38, 68.66, 53.46, 31.96, 10.30, 2.14],  #12
                               [58.68, 56.24, 52.94, 42.14, 24.60, 7.16, 1.58],  #6
                               [25.92, 23.78, 21.52, 17.26, 10.18, 3.46, 0.98],  #0
                               [5.38, 5.20, 5.24, 4.76, 2.94, 1.70, 0.50],  #-6
                               [1.62, 1.64, 1.46, 1.30, 1.22, 0.68, 0.22],  #-12
                               [0.98, 0.70, 0.76, 0.66, 0.68, 0.24, 0.18]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_256_capret_r10'] = [[88.24, 85.14, 82.24, 71.66, 47.06, 16.12, 3.18],  #18
                               [85.18, 83.26, 79.30, 67.80, 44.08, 15.44, 3.44],  #12
                               [71.78, 70.26, 66.44, 55.28, 34.70, 11.12, 3.12],  #6
                               [36.34, 33.62, 31.12, 26.14, 16.06, 5.56, 1.86],  #0
                               [8.96, 8.62, 8.88, 7.50, 4.78, 2.98, 0.88],  #-6
                               [3.04, 2.74, 2.30, 2.38, 1.92, 1.16, 0.34],  #-12
                               [1.66, 1.46, 1.44, 1.30, 1.10, 0.66, 0.36]]   #-18
################################################################### 16 128 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_imgret_r1'] = [[35.18, 33.95, 32.33, 25.96, 12.44, 2.31, 0.63],  #18
                               [28.99, 28.34, 26.92, 21.52, 11.08, 2.28, 0.36],  #12
                               [16.82, 18.18, 17.04, 13.68, 7.07, 1.83, 0.43],  #6
                               [6.22, 6.23, 5.62, 4.70, 2.98, 0.94, 0.44],  #0
                               [1.54, 1.50, 1.60, 1.18, 0.83, 0.42, 0.18],  #-6
                               [0.44, 0.56, 0.54, 0.37, 0.24, 0.24, 0.12],  #-12
                               [0.27, 0.22, 0.12, 0.23, 0.17, 0.09, 0.10]]   #-18

###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_imgret_r5'] = [[69.22, 68.69, 65.95, 57.98, 35.66, 9.32, 2.32],  #18
                               [62.08, 61.89, 60.22, 52.92, 32.42, 9.03, 2.56],  #12
                               [43.73, 44.61, 43.59, 38.20, 22.71, 7.52, 2.23],  #6
                               [19.48, 20.29, 18.48, 16.18, 10.57, 4.70, 1.77],  #0
                               [5.85, 5.94, 5.74, 5.07, 3.78, 1.97, 0.98],  #-6
                               [2.20, 2.10, 2.13, 1.89, 1.16, 1.09, 0.62],  #-12
                               [1.24, 1.06, 0.80, 0.92, 1.01, 0.53, 0.54]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_imgret_r10'] = [[81.09, 80.08, 78.01, 71.40, 49.56, 15.83, 4.29],  #18
                               [75.36, 75.04, 73.83, 67.18, 45.88, 15.05, 4.96],  #12
                               [58.59, 59.48, 58.09, 52.63, 34.18, 13.22, 4.28],  #6
                               [30.08, 30.97, 28.32, 25.32, 17.57, 9.06, 3.54],  #0
                               [10.46, 10.31, 9.95, 9.14, 6.94, 3.87, 1.85],  #-6
                               [4.00, 3.82, 3.88, 3.40, 2.14, 1.99, 1.14],  #-12
                               [2.33, 2.02, 1.94, 1.83, 1.93, 1.01, 1.03]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_capret_r1'] = [[45.76, 44.64, 36.18, 24.28, 10.48, 2.38, 0.48],  #18
                               [41.76, 39.94, 33.08, 22.62, 9.96, 2.18, 0.82],  #12
                               [27.04, 27.64, 24.78, 16.48, 6.84, 2.02, 0.54],  #6
                               [8.72, 9.38, 7.46, 6.04, 3.04, 1.40, 0.30],  #0
                               [1.82, 1.38, 1.36, 1.30, 0.78, 0.50, 0.22],  #-6
                               [0.42, 0.40, 0.40, 0.46, 0.24, 0.32, 0.18],  #-12
                               [0.28, 0.22, 0.16, 0.20, 0.16, 0.06, 0.08]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_capret_r5'] = [[77.10, 74.52, 66.56, 47.34, 22.68, 5.28, 1.48],  #18
                               [72.60, 70.22, 62.46, 44.42, 21.26, 5.08, 1.38],  #12
                               [56.14, 55.34, 49.32, 36.22, 15.38, 4.44, 1.16],  #6
                               [23.28, 23.54, 20.04, 14.94, 7.88, 3.16, 0.90],  #0
                               [5.16, 5.24, 4.98, 4.26, 2.52, 1.36, 0.60],  #-6
                               [1.56, 1.44, 1.52, 1.26, 0.74, 0.62, 0.38],  #-12
                               [0.92, 0.94, 0.64, 0.66, 0.44, 0.12, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_128_capret_r10'] = [[87.26, 84.98, 78.52, 61.36, 33.20, 8.56, 2.30],  #18
                               [83.52, 81.76, 75.28, 58.18, 30.40, 8.10, 2.34],  #12
                               [69.14, 68.76, 62.62, 47.92, 23.04, 7.86, 2.04],  #6
                               [33.72, 34.10, 28.86, 22.34, 12.22, 4.82, 1.52],  #0
                               [8.34, 8.82, 7.92, 6.76, 4.14, 2.30, 0.98],  #-6
                               [2.90, 2.66, 2.68, 2.48, 1.26, 1.04, 0.52],  #-12
                               [1.74, 1.50, 1.48, 1.18, 0.76, 0.24, 0.34]]   #-18
################################################################### 16 64 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_imgret_r1'] = [[34.73, 33.88, 29.63, 20.26, 6.83, 1.19, 0.39],  #18
                               [28.41, 27.57, 24.12, 16.28, 5.55, 1.16, 0.39],  #12
                               [17.36, 17.51, 15.47, 11.04, 4.14, 1.00, 0.40],  #6
                               [5.70, 6.31, 5.36, 3.59, 1.44, 0.49, 0.30],  #0
                               [1.27, 1.37, 1.20, 1.04, 0.55, 0.26, 0.18],  #-6
                               [0.52, 0.44, 0.39, 0.31, 0.34, 0.24, 0.07],  #-12
                               [0.25, 0.23, 0.24, 0.18, 0.22, 0.11, 0.09]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_imgret_r5'] = [[68.50, 67.38, 62.71, 50.03, 21.36, 5.18, 1.78],  #18
                               [61.66, 60.48, 56.20, 43.40, 19.06, 5.11, 1.79],  #12
                               [44.40, 44.58, 40.52, 31.78, 14.36, 4.21, 1.70],  #6
                               [19.13, 19.50, 17.58, 13.16, 6.62, 2.58, 1.21],  #0
                               [5.93, 5.38, 5.25, 4.51, 2.60, 0.99, 0.86],  #-6
                               [2.10, 1.77, 1.98, 1.64, 1.29, 0.92, 0.43],  #-12
                               [0.95, 1.18, 0.96, 0.98, 0.86, 0.41, 0.45]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_imgret_r10'] = [[80.28, 79.32, 75.53, 63.59, 32.92, 9.06, 3.38],  #18
                               [75.48, 74.08, 70.30, 57.76, 29.33, 9.20, 3.46],  #12
                               [59.20, 59.72, 54.90, 45.87, 22.72, 7.43, 3.29],  #6
                               [29.43, 29.70, 27.48, 21.50, 11.71, 4.98, 2.67],  #0
                               [10.27, 9.36, 9.50, 8.12, 4.82, 2.00, 1.80],  #-6
                               [3.81, 3.66, 3.64, 3.20, 2.99, 1.97, 0.94],  #-12
                               [1.94, 2.00, 1.79, 1.86, 1.59, 0.89,  0.98]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_capret_r1'] = [[44.86, 40.04, 30.36, 17.52, 5.70, 1.52, 0.36],  #18
                               [41.68, 35.16, 27.96, 15.70, 5.06, 1.30, 0.42],  #12
                               [26.80, 26.52, 20.64, 11.98, 4.46, 1.02, 0.40],  #6
                               [8.54, 8.64, 6.28, 3.98, 1.46, 0.64, 0.26],  #0
                               [1.56, 1.46, 1.82, 0.84, 0.44, 0.18, 0.24],  #-6
                               [0.48, 0.26, 0.40, 0.38, 0.22, 0.18, 0.06],  #-12
                               [0.20, 0.14, 0.18, 0.16, 0.12, 0.04, 0.14]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_capret_r5'] = [[74.94, 71.52, 57.98, 36.06, 12.16, 3.06, 0.84],  #18
                               [71.64, 66.28, 54.56, 33.72, 11.00, 2.76, 0.76],  #12
                               [55.58, 52.44, 42.26, 25.58, 9.42, 2.42, 0.80],  #6
                               [23.00, 22.30, 17.72, 10.56, 4.24, 1.50, 0.64],  #0
                               [5.26, 5.00, 4.82, 3.14, 1.60, 0.58, 0.66],  #-6
                               [1.88, 1.30, 1.28, 1.18, 0.72, 0.66, 0.20],  #-12
                               [0.98, 0.66, 0.62, 0.80, 0.40, 0.24, 0.34]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_64_capret_r10'] = [[85.60, 82.50, 70.96, 48.68, 18.42, 4.96, 1.56],  #18
                               [82.70, 78.98, 67.40, 46.14, 16.54, 4.82, 1.48],  #12
                               [68.28, 66.28, 55.16, 36.28, 14.48, 3.92, 1.60],  #6
                               [32.26, 32.50, 26.50, 16.38, 6.90, 2.56, 1.06],  #0
                               [8.72, 8.18, 7.56, 5.24, 2.70, 1.04, 0.94],  #-6
                               [2.96, 2.52, 2.36, 1.78, 1.46, 1.02, 0.40],  #-12
                               [1.86, 1.24, 1.20, 1.38, 0.74, 0.50, 0.62]]   #-18
################################################################### 16 32 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_imgret_r1'] = [[34.40, 32.40, 24.95, 12.76, 3.13, 0.74, 0.45],  #18
                               [27.82, 26.32, 21.64, 10.92, 2.42, 0.61, 0.18],  #12
                               [17.31, 16.50, 13.27, 6.48, 1.83, 0.51, 0.25],  #6
                               [5.88, 5.66, 4.51, 2.57, 1.26, 0.38, 0.23],  #0
                               [1.59, 1.19, 1.30, 0.94, 0.61, 0.20, 0.17],  #-6
                               [0.64, 0.45, 0.46, 0.20, 0.22, 0.14, 0.06],  #-12
                               [0.21, 0.21, 0.23, 0.22, 0.06, 0.10, 0.05]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_imgret_r5'] = [[68.60, 66.87, 56.92, 35.87, 11.58, 3.13, 1.48],  #18
                               [61.05, 58.61, 52.32, 33.02, 9.31, 2.85, 1.12],  #12
                               [44.45, 42.79, 37.45, 22.21, 7.79, 2.13, 1.10],  #6
                               [19.53, 18.48, 15.62, 9.61, 4.87, 1.70, 0.81],  #0
                               [6.06, 5.00, 5.13, 3.56, 2.50, 0.97, 0.56],  #-6
                               [2.39, 1.91, 1.82, 1.03, 1.00, 0.97, 0.41],  #-12
                               [1.04, 1.06, 0.91, 0.99, 0.43, 0.55, 0.40]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_imgret_r10'] = [[80.76, 79.13, 70.80, 49.51, 19.00, 5.63, 2.67],  #18
                               [74.46, 71.94, 66.82, 46.41, 15.97, 5.50, 2.12],  #12
                               [59.21, 57.53, 51.40, 34.06, 13.43, 4.13, 2.11],  #6
                               [30.02, 28.22, 24.92, 16.15, 8.67, 3.23, 1.81],  #0
                               [10.22, 9.09, 9.07, 6.81, 4.35, 2.08, 1.03],  #-6
                               [4.55, 3.56, 3.54, 1.98, 1.93, 1.96, 0.83],  #-12
                               [1.94, 2.05, 1.63, 1.73, 1.04, 1.13, 0.91]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_capret_r1'] = [[43.02, 36.96, 24.98, 10.64, 2.98, 0.78, 0.28],  #18
                               [38.70, 33.00, 23.32, 9.80, 2.26, 0.82, 0.28],  #12
                               [27.12, 23.50, 15.88, 6.60, 1.82, 0.54, 0.28],  #6
                               [8.94, 6.72, 4.86, 2.46, 1.28, 0.40, 0.18],  #0
                               [1.40, 1.32, 1.12, 0.76, 0.58, 0.24, 0.10],  #-6
                               [0.48, 0.44, 0.16, 0.16, 0.24, 0.16, 0.02],  #-12
                               [0.18, 0.26, 0.10, 0.22, 0.10, 0.08, 0.04]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_capret_r5'] = [[74.02, 66.66, 47.82, 22.52, 6.54, 1.54, 0.86],  #18
                               [69.36, 60.44, 46.18, 21.38, 5.32, 1.58, 0.58],  #12
                               [53.90, 47.84, 34.68, 16.00, 4.64, 1.12, 0.62],  #6
                               [22.46, 19.06, 13.92, 6.58, 2.80, 0.98, 0.44],  #0
                               [5.26, 4.60, 3.96, 2.38, 1.68, 0.68, 0.44],  #-6
                               [1.54, 1.30, 0.90, 0.74, 0.70, 0.50, 0.24],  #-12
                               [0.86, 0.90, 0.66, 0.66, 0.28, 0.50, 0.42]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_32_capret_r10'] = [[85.20, 78.84, 61.18, 32.18, 9.60, 2.94, 1.42],  #18
                               [80.54, 73.84, 58.98, 31.24, 8.82, 2.88, 1.14],  #12
                               [67.24, 61.20, 46.50, 23.82, 7.40, 2.02, 0.90],  #6
                               [32.94, 28.08, 21.48, 10.88, 4.92, 1.56, 0.88],  #0
                               [8.46, 7.38, 6.68, 4.10, 2.54, 1.22, 0.58],  #-6
                               [2.94, 2.22, 1.86, 1.38, 1.14, 0.76, 0.46],  #-12
                               [1.70, 1.46, 1.24, 1.08, 0.48, 0.76, 0.68]]   #-18
################################################################### 16 16 #######################################################################################
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_imgret_r1'] = [[30.39, 27.56, 17.66, 6.55, 1.43, 0.40, 0.32],  #18
                               [24.51, 23.78, 16.21, 5.25, 1.42, 0.26, 0.32],  #12
                               [14.64, 13.85, 7.91, 3.58, 0.71, 0.30, 0.37],  #6
                               [5.71, 4.61, 3.66, 1.52, 0.58, 0.42, 0.24],  #0
                               [1.13, 1.29, 1.04, 0.65, 0.24, 0.35, 0.29],  #-6
                               [0.39, 0.53, 0.38, 0.22, 0.18, 0.25, 0.21],  #-12
                               [0.26, 0.24, 0.22, 0.19, 0.26, 0.42, 0.34]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_imgret_r5'] = [[64.27, 60.38, 46.03, 22.07, 6.08, 1.74, 1.05],  #18
                               [57.35, 55.35, 44.08, 18.12, 6.18, 1.47, 1.39],  #12
                               [40.45, 37.56, 25.22, 14.51, 3.30, 1.57, 1.38],  #6
                               [19.13, 15.76, 13.19, 6.38, 2.34, 1.71, 1.32],  #0
                               [5.28, 4.74, 4.28, 3.34, 1.43, 1.65, 1.33],  #-6
                               [1.93, 2.19, 1.58, 1.10, 1.16, 1.25, 1.26],  #-12
                               [1.17, 1.04, 1.08, 1.19, 1.45, 1.93, 1.44]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_imgret_r10'] = [[77.50, 73.94, 60.66, 33.14, 11.30, 3.66, 2.02],  #18
                               [71.58, 69.35, 58.85, 28.57, 10.53, 2.98, 2.50],  #12
                               [55.08, 51.46, 38.56, 23.50, 6.09, 2.98, 2.46],  #6
                               [29.18, 25.14, 21.23, 11.52, 4.60, 3.16, 2.51],  #0
                               [9.40, 8.39, 7.51, 6.20, 3.06, 3.19, 2.66],  #-6
                               [3.85, 3.96, 3.22, 2.24, 2.28, 2.41, 2.44],  #-12
                               [2.20, 1.93, 2.20, 2.24, 2.81, 3.48, 2.81]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_capret_r1'] = [[37.20, 28.46, 15.22, 5.18, 1.46, 0.40, 0.20],  #18
                               [33.02, 27.14, 15.06, 4.72, 1.42, 0.34, 0.28],  #12
                               [22.30, 16.90, 9.08, 3.56, 0.78, 0.26, 0.26],  #6
                               [8.42, 5.78, 3.88, 1.58, 0.50, 0.28, 0.28],  #0
                               [1.02, 1.24, 0.88, 0.76, 0.26, 0.26, 0.34],  #-6
                               [0.40, 0.58, 0.44, 0.30, 0.24, 0.34, 0.26],  #-12
                               [0.24, 0.18, 0.16, 0.24, 0.18, 0.54, 0.32]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_capret_r5'] = [[67.20, 56.62, 33.04, 11.86, 3.50, 0.92, 0.50],  #18
                               [63.00, 53.48, 32.94, 10.48, 3.40, 0.78, 0.56],  #12
                               [47.94, 38.00, 21.04, 8.44, 1.92, 0.72, 0.68],  #6
                               [21.92, 16.00, 10.90, 4.40, 1.62, 0.64, 0.38],  #0
                               [4.50, 3.88, 2.34, 2.44, 0.74, 0.72, 0.72],  #-6
                               [1.48, 1.86, 1.42, 0.74, 0.62, 0.56, 0.42],  #-12
                               [1.08, 0.76, 0.62, 0.84, 0.68, 0.84, 0.80]]   #-18
###############################  18    12    6      0     -6    -12   -18
acc['symb16_16_capret_r10'] = [[79.18, 69.48, 45.78, 17.64, 5.68, 1.66, 0.86],  #18
                               [76.16, 66.66, 44.98, 15.68, 5.42, 1.36, 0.82],  #12
                               [61.38, 50.36, 31.04, 13.90, 3.30, 1.30, 1.26],  #6
                               [31.60, 23.82, 17.40, 6.86, 2.88, 1.18, 0.64],  #0
                               [7.42, 6.84, 4.24, 4.22, 1.50, 1.20, 1.40],  #-6
                               [2.68, 3.02, 2.22, 1.34, 1.10, 0.88, 0.80],  #-12
                               [1.86, 1.34, 1.22, 1.36, 1.18, 1.36, 1.46]]   #-18

############################################################ opt result
opt_acc = {}
opt_acc['amb_imgret_r1'] = [0.34,  0.42,  0.42,  1.52,  7.91, 23.78, 39.25, 53.96, 60.91, 65.82, 67.52]
opt_acc['amb_imgret_r5'] = [1.44,  1.93,  1.93,  6.38, 25.22, 55.35, 72.39, 82.85, 86.72, 88.92, 89.8]
opt_acc['amb_imgret_r10'] = [2.81,  3.48,  3.48, 11.52, 38.56, 69.35, 83.08, 89.98, 92.57, 93.86, 94.34]
opt_acc['amb_capret_r1'] = [0.32,  0.54,  0.54,  1.58,  9.08, 27.14, 47.54, 64.88, 75.84, 79.82, 82.04]
opt_acc['amb_capret_r5'] = [0.8,   0.84,  0.84,  4.4,  21.04, 53.48, 75.06, 88.02, 93.38, 94.98, 95.66]
opt_acc['amb_capret_r10'] = [1.46,  1.46,  1.5,   6.86, 31.04, 66.66, 84.8,  93.56, 96.82, 97.84, 98.22]

opt_acc['sca_imgret_r1'] = [ 0.34,  0.42,  0.42,  1.52,  7.91, 23.78, 39.25, 53.96, 60.91, 65.82, 67.52]
opt_acc['sca_imgret_r5'] = [ 1.44,  1.93,  1.93,  6.38, 25.22, 55.35, 72.39, 82.85, 86.72, 88.92, 89.8]
opt_acc['sca_imgret_r10'] = [ 2.81,  3.48,  3.48, 11.52, 38.56, 69.35, 83.08, 89.98, 92.57, 93.86, 94.34]
opt_acc['sca_capret_r1'] = [ 0.32,  0.54,  0.32,  1.58,  9.08, 27.14, 47.54, 64.88, 75.84, 79.82, 82.04]
opt_acc['sca_capret_r5'] = [ 0.8,   0.84,  0.74,  4.4,  21.04, 53.48, 75.06, 88.02, 93.38, 94.98, 95.66]
opt_acc['sca_capret_r10'] = [ 1.46,  1.46,  1.5,   6.86, 31.04, 66.66, 84.8,  93.56, 96.82, 97.84, 98.22]
# opt for img ret r@1
opt_ri_sca = [5, 5, 5, 8, 9, 10, 11, 12, 13, 14, 15]
opt_rc_sca = [5, 6, 6, 8, 9, 10, 11, 12, 13, 14, 15]
opt_ni_sca = [4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8]
opt_nc_sca = [4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8]






def get_acc():
    '''
    :return: a 4D matrix of accuracy value: [snr_img, snr_cap, numsym_img, numsym_cap]
    '''
    # init variables
    snr_cap = np.arange(18, -19, -6)  # cap snr
    snr_img = np.arange(18, -19, -6)  # img snr
    numsym_img = np.arange(4, 9, 1)  # img num symbol, 256 = 2**8
    numsym_cap = np.arange(4, 9, 1)  # cap num symbol, 256 = 2**8

    # init acc
    Z_acc = {}
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            Z_acc[f'{ic}ret_r{r}'] = np.zeros((len(snr_img), len(snr_cap), len(numsym_img), len(numsym_cap)))

    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            for i_ni, ni in enumerate(numsym_img):
                for i_nc, nc in enumerate(numsym_cap):
                    Z_acc[f'{ic}ret_r{r}'][:, :, i_ni, i_nc] = acc[f'symb{2 ** ni}_{2 ** nc}_{ic}ret_r{r}']

    return Z_acc

def get_res_from_sn(s, n, channel_snr):
    """
    :param s: snr of img, in dB
    :param n: num of symbol of img, in dB
    :param channel_snr: environment snr in dB
    :return: total res, img res, cap res in dB
    """
    res = (s - channel_snr) / 6 + n
    return res

def get_snr_from_res_numsym(channel_snr, res, numsym):
    """
    :param channel_snr: environment snr in dB
    :param res: in dB
    :param numsym: in dB
    :return: snr in dB
    """
    assert res >= numsym, 'res must >= num of symbol'
    return (res - numsym) * 6.0 + channel_snr

def transf_acc4D_snr2res(acc, channel_snr, snr_img, snr_cap, numsym_img, numsym_cap):
    '''
    Transform accuracy matrix: [snr_img, snr_cap, numsym_img, numsym_cap] into [res_img, res_cap, numsym_img, numsym_cap]
    :return: accuracy matrix:[res_img, res_cap, numsym_img, numsym_cap]
    '''
    acc4D_numsym_res = {}
    res_img = [get_res_from_sn(si, ni, channel_snr) for si in snr_img for ni in numsym_img]
    res_img = list(dict.fromkeys(res_img))
    res_img.sort()
    res_img = np.array(res_img)
    res_cap = [get_res_from_sn(sc, nc, channel_snr) for sc in snr_cap for nc in numsym_cap]
    res_cap = list(dict.fromkeys(res_cap))
    res_cap.sort()
    res_cap = np.array(res_cap)
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            acc4D_numsym_res[f'{ic}ret_r{r}'] = np.zeros((len(res_img), len(res_cap), len(numsym_img), len(numsym_cap)))
    for i_ni, ni in enumerate(numsym_img):
        for i_ri, ri in enumerate(res_img):
            for i_nc, nc in enumerate(numsym_cap):
                for i_rc, rc in enumerate(res_cap):
                    if ri < ni or rc < nc:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    si = get_snr_from_res_numsym(channel_snr, ri, ni)
                    if si not in snr_img:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    sc = get_snr_from_res_numsym(channel_snr, rc, nc)
                    if sc not in snr_cap:
                        for ic in ['img', 'cap']:
                            for r in [1, 5, 10]:
                                acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = np.nan
                        continue
                    for ic in ['img', 'cap']:
                        for r in [1, 5, 10]:
                            i_si = np.where(snr_img==si)[0]
                            i_sc = np.where(snr_cap==sc)[0]
                            acc4D_numsym_res[f'{ic}ret_r{r}'][i_ri, i_rc, i_ni, i_nc] = acc[f'{ic}ret_r{r}'][i_si, i_sc, i_ni, i_nc]

    return acc4D_numsym_res



####################################################################################################################################################################################

####################################################################################################################################################################################
if __name__ == '__main__':
    from utils.utils import groups_from_res_imgandcap, groups_from_res_imgorcap, get_opt_acc_from_imgres_capres, res_calculator


    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 10 colors
    ############################################################### noise one side
    SNR=[18, 12, 6, 0, -6, -12, -18]
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(16, 9)
    for j, ic in enumerate(['img', 'cap']):
        for i, r in enumerate([1, 5, 10]):
            noise_both = []
            for temp_i in range(7):
                noise_both.append(np.array(acc[f'symb256_256_{ic}ret_r{r}'])[temp_i, temp_i])


            ax[j, i].plot(SNR, acc[f'symb256_256_{ic}ret_r{r}_both'], 'r', linewidth=3.0, label='Scenario a')
            ax[j, i].plot(SNR, acc[f'symb256_256_{ic}ret_r{r}_ncap'], 'b--*', linewidth=3.0, markersize=12, label='Scenario b')
            ax[j, i].plot(SNR, acc[f'symb256_256_{ic}ret_r{r}_nimg'], 'b--o', linewidth=3.0, markersize=10, label='Scenario c')
            ax[j, i].plot(SNR, np.array(noise_both), 'b', linewidth=3.0, label='Scenario d')
            ax[j, i].set_xlabel('SNR/dB', fontsize=15)
            ax[j, i].set_ylabel('Accuracy/%', fontsize=15)
            ax[j, i].set_ylim(0, 100)
            ax[j, i].grid()
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

    ################################################# k leyer AE
    # k_layers_AE
    SNR_AE = [18, 6, -6, -18]
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(16, 9)
    for j, ic in enumerate(['img', 'cap']):
        for i, r in enumerate([1, 5, 10]):
            for k in range(1, 4, 1):
                ax[j, i].plot(SNR_AE, acc_k[f'k{k}_{ic}ret_r{r}'], color=colors[k-1], marker='*', markersize=15, linewidth=3.0, label=f'{k} layers AutoEncoder')
                ax[j, i].set_xlabel('SNR', fontsize=20)
                ax[j, i].set_ylabel('Accuracy', fontsize=20)
                ax[j, i].grid()
                ax[j, i].set_ylim(0, 100)
    lines_labels = [ax[0, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, fontsize='xx-large')
    plt.show()


    SNR_range = [18, 12, 6, 0, -6, -12, -18]
    max_num_symbol=256
    min_num_symbol=16
    max_SNR = 18
    min_SNR = -18
    snr_img = np.arange(18, -19, -6)  # img snr
    snr_cap = np.arange(18, -19, -6)  # cap snr
    numsym_img = np.arange(4, 9, 1)  # number of symbol img
    numsym_cap = np.arange(4, 9, 1)  # number of symbol cap
    channel_SNR = -24

    ################################## fix ri, ni, projection   ############ x_res_range_db, y_res_in_total
    # S1_proj_with_black_line_text
    acc_sn = get_acc()
    acc_nr = transf_acc4D_snr2res(acc_sn, channel_SNR, snr_img, snr_cap, numsym_img, numsym_cap)
    colors = ['blue', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1),
    #           (0.2, 0.4, 0.7), (0.2, 0.7, 0.4), (0.7, 0.2, 0.4), (0.7, 0.4, 0.2), (0.4, 0.7, 0.2), (0.4, 0.2, 0.7),
    #           (0.1, 0.1, 0.8), (0.8, 0.1, 0.1), (0.1, 0.8, 0.1)]
    res_range_db = np.arange(5, 16, 1)
    fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
    fig.set_size_inches(16, 9)
    for j, ic in enumerate(['img', 'cap']):
        for i, r in enumerate([1, 5, 10]):
            acc_nr_2D = acc_nr[f'{ic}ret_r{r}'][8, :, 2, :]
            for i_res_db, res_db in enumerate(res_range_db):
                z_acc = acc_nr_2D[i_res_db, :]
                mask_z = ~np.isnan(z_acc)
                z_acc = z_acc[mask_z]
                x_num_symbol = numsym_cap[mask_z]
                y_res = np.array([res_db] * x_num_symbol.shape[0])
                ax[j, i].plot3D(x_num_symbol, y_res, z_acc, color=colors[i_res_db], label=f'res = 2^{res_db}')
                ######## add projection
                z_null = np.array([0] * x_num_symbol.shape[0])
                ax[j, i].plot3D(x_num_symbol, y_res, z_null, color=colors[i_res_db], linestyle='dashed')
                y = np.array([res_db] * x_num_symbol.shape[0])
                y = np.concatenate((y, y), axis=0)
                X, Y = np.meshgrid(x_num_symbol, y)
                Z = np.repeat(np.concatenate((np.expand_dims(z_acc, axis=0), np.expand_dims(z_null, axis=0)), axis=0), x_num_symbol.shape[0], axis=0)
                ax[j, i].plot_surface(X, Y, Z, color=colors[i_res_db], alpha=0.1)

            ######## add lines with number
            for i_x, x in enumerate(numsym_cap):
                z_line = acc_nr_2D[:, i_x]
                mask_z = ~np.isnan(z_line)
                z_line = z_line[mask_z]
                y_line = res_range_db[mask_z]
                x_line = np.array([x] * y_line.shape[0])
                ax[j, i].plot3D(x_line, y_line, z_line, color='k')
                for i_text, text in enumerate(z_line):
                    x_pos = x_line[i_text]
                    y_pos = y_line[i_text]
                    ax[j, i].text(x_pos, y_pos, text, '%.2f' % text, fontdict={'fontsize': 7})
                # if len(x_line) >= 2:
                #     ax[j, i].text(x_line, y_line, z_line, '%.2f' % z_line, fontdict={'fontsize':14})

            ax[j, i].set_xlabel('$\mathbf{d^{cap}_i}$', fontsize=12, labelpad=10)
            ax[j, i].set_ylabel('$\mathbf{B^{cap}_i}$', fontsize=12, labelpad=10)
            ax[j, i].set_xticks(range(4, 9, 1))
            ax[j, i].set_xticklabels([f'{i}' for i in range(4, 9, 1)])
            y_res_range_db = np.array(res_range_db)
            ax[j, i].set_yticks(y_res_range_db[1::3])
            ax[j, i].set_yticklabels([f'2^{res_db}' for res_db in y_res_range_db[0::3]])
            ax[j, i].set_zlabel('Accuracy', fontsize=10)
            ax[j, i].set_zlim(0, 100)
            ax[j, i].view_init(20, 320, None)
    ax[0, 0].annotate('R@1', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[0, 1].annotate('R@5', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[0, 2].annotate('R@10', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[0, 0].annotate('retrieval', xy=(-0.2, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[0, 0].annotate('Image', xy=(-0.2, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[1, 0].annotate('retrieval', xy=(-0.2, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    ax[1, 0].annotate('Caption', xy=(-0.2, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



                # # ################################## S1_fixRes_3D.png   ############ x_num_symbol, y_res, z_acc
    # ################################## S1_fixRes_maxAcc.png   ############ x_res_range_db, y_res_in_total
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1),
              (0.2, 0.4, 0.7), (0.2, 0.7, 0.4), (0.7, 0.2, 0.4), (0.7, 0.4, 0.2), (0.4, 0.7, 0.2), (0.4, 0.2, 0.7),
              (0.1, 0.1, 0.8), (0.8, 0.1, 0.1), (0.1, 0.8, 0.1)]
    res_range_db = np.arange(4, 15, 1)
    # colors = ['blue', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # res_range_db = np.arange(7, 8, 1)
    max_acc={}
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
            max_acc[f'{ic}ret_r{r}']=[]
    fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
    fig.set_size_inches(16, 9)
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
            for i_res_db, res_db in enumerate(res_range_db):
                groups = np.array(groups_from_res_imgorcap(2**res_db, -24))
                if groups.shape[0] > 0:
                    groups = groups[groups[:, 0] <= max_num_symbol]
                    groups = groups[groups[:, 0] >= min_num_symbol]
                    groups = groups[groups[:, 1] <= max_SNR]
                    groups = groups[groups[:, 1] >= min_SNR]
                    i_groups = []
                    for i_group, group in enumerate(groups):
                        num_symbol, snr = group[0], group[1]
                        # aa=math.log2(res_calculator(num_symbol, snr, channel_SNR))
                        # print(aa)
                        if abs(math.log2(res_calculator(num_symbol, snr, channel_SNR)) - res_db) > 0.1:
                            i_groups.append(i_group)
                            # groups = np.delete(groups, (i_group), axis=0)
                    groups = np.delete(groups, (i_groups), axis=0)
                    # groups = groups[abs(math.log2(res_calculator(groups[:, 0], groups[:, 1], channel_SNR))-res_db) < 0.1 ]
                    if groups.shape[0] >= 1:
                        groups = list(groups)
                        z_acc = [acc[f'symb{num_symbol}_{num_symbol}_{ic}ret_r{r}'][SNR_range.index(snr)][SNR_range.index(snr)] for num_symbol, snr in groups]
                        x_num_symbol = np.array([num_symbol * 2 for num_symbol, snr in groups])
                        y_res = np.array([res_db] * x_num_symbol.shape[0])
                        ax[j, i].plot3D(x_num_symbol, y_res, z_acc, color=colors[i_res_db], label=f'res = 2^{res_db}')
                        ######## add projection
                        z_null = np.array([0] * x_num_symbol.shape[0])
                        ax[j, i].plot3D(x_num_symbol, y_res, z_null, color=colors[i_res_db], linestyle='dashed')
                        y = np.array([res_db] * x_num_symbol.shape[0])
                        y = np.concatenate((y, y), axis=0)
                        X, Y = np.meshgrid(x_num_symbol, y)
                        Z = np.repeat(np.concatenate((np.expand_dims(z_acc, axis=0), np.expand_dims(z_null, axis=0)), axis=0), x_num_symbol.shape[0], axis=0)
                        ax[j, i].plot_surface(X, Y, Z, color=colors[i_res_db], alpha=0.1)

                        ax[j, i].set_xlabel('num of symbol img or cap')
                        ax[j, i].set_ylabel('res one side')
                        ax[j, i].set_yticks(np.array(res_range_db))
                        ax[j, i].set_yticklabels([f'2^{res_db}' for res_db in res_range_db])
                        ax[j, i].set_zlabel('acc')
                        ax[j, i].legend()

                        max_acc[f'{ic}ret_r{r}'].append(max(z_acc))
                    else:
                        max_acc[f'{ic}ret_r{r}'].append(None)
                else:
                    max_acc[f'{ic}ret_r{r}'].append(None)
    plt.show()

    fig, ax = plt.subplots()
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
            ax.plot(np.array(res_range_db), max_acc[f'{ic}ret_r{r}'], label=f'{ic}ret_r{r}')
    ax.set_xlabel('res one side')
    ax.set_ylabel('opt acc')
    ax.set_xticks(np.array(res_range_db))
    ax.set_xticklabels([f'2^{res_db}' for res_db in res_range_db])
    ax.legend(loc='upper left',fontsize='small')
    plt.show()



    ############################################# S3_resimg_rescap_optacc.png
    # S3_resimg_rescap_optacc
    res_oneside_range_db = np.arange(4, 15, 1)
    X, Y = np.meshgrid(res_oneside_range_db, res_oneside_range_db) # X:cap, Y: img
    Z = {}
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
             Z[f'{ic}ret_r{r}'] = np.zeros(X.shape) # Z.shape() = (img, cap)
    fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={"projection": "3d"})
    fig.set_size_inches(16, 9)
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
            for i_img_res, img_res in enumerate(res_oneside_range_db):
                for i_cap_res, cap_res in enumerate(res_oneside_range_db):
                    Z[f'{ic}ret_r{r}'][i_img_res, i_cap_res] = get_opt_acc_from_imgres_capres(acc, img_res, cap_res, f'{ic}ret_r{r}', SNR_range, max_num_symbol, min_num_symbol, max_SNR, min_SNR)
            surf = ax[j, i].plot_surface(X, Y, Z[f'{ic}ret_r{r}'], cmap='YlOrRd', linewidth=0, antialiased=False)
            ax[j, i].set_xlabel('$\mathbf{B^{img}_i}$', fontsize=10, labelpad=8)
            ax[j, i].set_ylabel('$\mathbf{B^{cap}_i}$', fontsize=10, labelpad=8)
            ax[j, i].set_zlabel('Optimal accuracy', fontsize=10)
            temp_res_oneside_range_db = np.array(res_oneside_range_db)
            ax[j, i].set_xticks(temp_res_oneside_range_db[1::3])
            ax[j, i].set_yticks(temp_res_oneside_range_db[1::3])
            ax[j, i].set_xticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
            ax[j, i].set_yticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
            ax[j, i].set_zlim(0, 100)
            ax[j, i].view_init(20, 300, None)
            ax[j, i].grid()

    ax[0, 0].annotate('R@1', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[0, 1].annotate('R@5', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[0, 2].annotate('R@10', xy=(0.5, 0.9), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[0, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[0, 0].annotate('Image', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[1, 0].annotate('retrieval', xy=(-0.1, 0.5), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    ax[1, 0].annotate('Caption', xy=(-0.1, 0.6), xytext=(5, 0), xycoords='axes fraction', textcoords='offset points', size='xx-large', ha='center', va='baseline')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    #############################################
    total_res_range = np.arange(5, 17, 1)
    opt_acc_totalres = {}
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
            opt_acc_totalres[f'{ic}ret_r{r}'] = np.zeros(total_res_range.shape)
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
            for i_total_res, total_res in enumerate(total_res_range):
                temp_max_acc = 0
                for img_res in np.arange(4, total_res, 1):
                    cap_res_db_upperbound = int(np.log2(2**total_res-2**img_res))
                    if cap_res_db_upperbound < 4: continue
                    cap_res_range = np.arange(4, cap_res_db_upperbound + 1, 1)
                    for cap_res in cap_res_range:
                        temp_max_acc = max(temp_max_acc, Z[f'{ic}ret_r{r}'][np.where(res_oneside_range_db == cap_res)[0], np.where(res_oneside_range_db == img_res)[0]])
                        # print('%d, %d' % (img_res, cap_res))
                opt_acc_totalres[f'{ic}ret_r{r}'][i_total_res] = temp_max_acc

    fig = plt.figure()
    ax = fig.add_subplot()
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
            ax.plot(total_res_range, opt_acc_totalres[f'{ic}ret_r{r}'], label=f'{ic}ret_r{r}')
            print(opt_acc_totalres[f'{ic}ret_r{r}'])
    ax.set_xlabel('total res')
    ax.set_ylabel('opt acc')
    ax.legend(loc='upper left',fontsize='small')
    xticks_labels=[f'2^{res_db}' for res_db in total_res_range]
    xticks(total_res_range, xticks_labels)
    plt.show()


    ############################################## S2 resimg = rescap but num symbols can be uneuqal for img and cap
    res_oneside_range_db = np.arange(5, 17, 1)
    opt_acc = np.zeros(res_oneside_range_db.shape)
    fig = plt.figure()
    ax = fig.add_subplot()
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
            for i_res, res in enumerate(res_oneside_range_db):
                opt_acc[i_res] = get_opt_acc_from_imgres_capres(acc, res, res, f'{ic}ret_r{r}', SNR_range)
            ax.plot(res_oneside_range_db, opt_acc, label=f'{ic}ret_r{r}')
    ax.set_xlabel('total res')
    ax.set_ylabel('opt acc')
    ax.legend(loc='upper left',fontsize='small')
    ax.set_xticks(np.array(res_oneside_range_db))
    ax.set_xticklabels([f'2^{res_db}' for res_db in res_oneside_range_db])
    plt.show()



    d1 = 16
    d2 = 256
    fig, ax = plt.subplots(nrows=2, ncols=3)
    for j, ic in enumerate(['img', 'cap']):
         for i, r in enumerate([1, 5, 10]):
             ax[j, i].plot(SNR_range, acc[f'symb_{d1}_{d2}_{ic}ret_r{r}'], linestyle='solid', color='red', label='img {d1} cap {d2}')
    ax[0, 0].set_xlabel('img and cap SNR')
    ax[0, 0].set_ylabel('acc/%')
    ax[0, 0].set_title('image retrieval R@1')
    ax[0, 0].legend(loc='upper left', fontsize='small')
    ax[0, 0].grid()

    fig.suptitle('acc when img and cap transmitted in different size')
    plt.show()

    total_res_range=np.arange(6, 17, 1)
    fig = plt.figure()
    ax = fig.add_subplot()
    i=0
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
            ax.plot(total_res_range, opt_acc[f'amb_{ic}ret_r{r}'], linestyle='solid', color=colors[i], label=f'exh_{ic}ret_r{r}')
            ax.plot(total_res_range, opt_acc[f'sca_{ic}ret_r{r}'], linestyle=':', marker='*', color=colors[i], label=f'sca_{ic}ret_r{r}')
            i += 1
    ax.set_xlabel('total res')
    ax.set_ylabel('opt acc')
    ax.legend(loc='upper left',fontsize='small')
    xticks_labels=[f'2^{res_db}' for res_db in total_res_range]
    xticks(total_res_range, xticks_labels)
    plt.show()

    ############################################## left_resimg_rescap_optacc, right_totalres_optacc, at img ret R@1
    # optacc_imgret_r1_left_right
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    res_oneside_range_db = np.arange(4, 16, 1)
    total_res_range = np.arange(5, 16, 1)
    X, Y = np.meshgrid(res_oneside_range_db, res_oneside_range_db)  # X:cap, Y: img
    Z = {}
    Z = np.zeros(X.shape)  # Z.shape() = (img, cap)
    for i_img_res, img_res in enumerate(res_oneside_range_db):
        for i_cap_res, cap_res in enumerate(res_oneside_range_db):
            # if img_res == 5 and cap_res == 5:
            #     a=1
            Z[i_img_res, i_cap_res] = get_opt_acc_from_imgres_capres(acc, img_res, cap_res, f'imgret_r1', SNR_range, max_num_symbol, min_num_symbol, max_SNR, min_SNR)
    # fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
    fig=plt.figure()
    fig.set_size_inches(16, 9)
    ax1=fig.add_subplot(1, 3, 1,  projection="3d")
    ax2=fig.add_subplot(1, 3, 3)
    ax3=fig.add_subplot(1, 3, 2,  projection="3d")

    ax1.plot_surface(X, Y, Z, cmap='YlOrRd', linewidth=0, antialiased=False, alpha=0.8)
    ax1.plot(opt_ri_sca, opt_rc_sca, opt_acc['sca_imgret_r1'], linestyle='solid',  marker='*', markersize=15, color=colors[0], linewidth=3.0)
    ax3.plot(opt_ni_sca, opt_nc_sca, opt_acc['sca_imgret_r1'], linestyle='solid',  marker='*', markersize=15, color=colors[0], linewidth=3.0)
    ax1.set_xlabel('$\mathbf{B^{img}_i}$', fontsize=12)
    ax1.set_ylabel('$\mathbf{B^{cap}_i}$', fontsize=12)
    ax3.set_xlabel('$\mathbf{d^{img}_i}$', fontsize=12)
    ax3.set_ylabel('$\mathbf{d^{cap}_i}$', fontsize=12)
    ax1.set_zlabel('Optimal accuracy', fontsize=10)
    temp_res_oneside_range_db = np.array(res_oneside_range_db)
    ax1.set_xticks(temp_res_oneside_range_db[1::3])
    ax1.set_yticks(temp_res_oneside_range_db[1::3])
    ax1.set_xticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_yticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_zlim(0, 100)
    ax1.view_init(40, 320, None)
    ax1.grid()
    i=0
    ax2.plot(total_res_range, opt_acc[f'amb_imgret_r1'], linestyle='solid', color=colors[i], linewidth=3.0, label=f'exh, r@{1}')
    ax2.plot(total_res_range, opt_acc[f'sca_imgret_r1'], marker='*', markersize=15, color=colors[i], linewidth=0, label=f'sca, r@{1}')
    i += 1
    for ic in ['img', 'cap']:
         for r in [1, 5, 10]:
             if ic!='img' or r!=1:
                ax2.plot(total_res_range, opt_acc[f'amb_{ic}ret_r{r}'], linestyle='solid', color=colors[i], linewidth=3.0, label=f'exh, r@{r}', alpha=0.4)
                ax2.plot(total_res_range, opt_acc[f'sca_{ic}ret_r{r}'], marker='*', markersize=15, color=colors[i], linewidth=0, label=f'sca, r@{r}', alpha=0.4)
                i += 1
    ax2.set_xlabel(r'$\mathbf{\bar{B}}$', fontsize=12)
    ax2.set_ylabel('Optimal accuracy', fontsize=10)
    ax2.legend(loc='upper left',fontsize='large')
    ax2.set_xticks(temp_res_oneside_range_db[1::3])
    ax2.set_xticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_position([0.05,0,0.25,1])
    ax2.set_position([0.65,0.38,0.25,0.25])
    ax3.set_position([0.34,0,0.25,1])
    # xticks_labels=[f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]]
    # xticks(temp_res_oneside_range_db[1::3], xticks_labels)
    ax2.set_ylim(0, 100)
    plt.show()
    ############################################## left_resimg_rescap_optacc, right_totalres_optacc, at img ret R@1
    # optacc_imgret_r1_left_right_coco
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    res_oneside_range_db = np.arange(4, 16, 1)
    total_res_range = np.arange(5, 16, 1)
    X, Y = np.meshgrid(res_oneside_range_db, res_oneside_range_db)  # X:cap, Y: img
    Z = {}
    Z = np.zeros(X.shape)  # Z.shape() = (img, cap)
    for i_img_res, img_res in enumerate(res_oneside_range_db):
        for i_cap_res, cap_res in enumerate(res_oneside_range_db):
            # if img_res == 5 and cap_res == 5:
            #     a=1
            Z[i_img_res, i_cap_res] = get_opt_acc_from_imgres_capres(acc, img_res, cap_res, f'imgret_r1', SNR_range, max_num_symbol, min_num_symbol, max_SNR, min_SNR)
    # fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
    fig=plt.figure()
    fig.set_size_inches(16, 9)
    ax1=fig.add_subplot(1, 2, 1,  projection="3d")
    ax2=fig.add_subplot(1, 2, 2)

    ax1.plot_surface(X, Y, Z, cmap='YlOrRd', linewidth=0, antialiased=False, alpha=0.8)
    ax1.plot(opt_ri_sca, opt_rc_sca, opt_acc['sca_imgret_r1'], linestyle='solid',  marker='*', markersize=15, color=colors[0], linewidth=3.0)
    ax1.set_xlabel('$\mathbf{B^{img}_i}$', fontsize=20)
    ax1.set_ylabel('$\mathbf{B^{cap}_i}$', fontsize=20)
    # ax1.set_zlabel('Optimal accuracy', fontsize=15)
    temp_res_oneside_range_db = np.array(res_oneside_range_db)
    ax1.set_xticks(temp_res_oneside_range_db[1::3])
    ax1.set_yticks(temp_res_oneside_range_db[1::3])
    ax1.set_xticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_yticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_zlim(0, 100)
    ax1.view_init(40, 320, None)
    ax1.grid()
    i=0
    ax2.plot(total_res_range, opt_acc[f'amb_imgret_r1'], linestyle='solid', color=colors[i], linewidth=3.0, label=f'exh, r@{1}')
    ax2.plot(total_res_range, opt_acc[f'sca_imgret_r1'], marker='*', markersize=15, color=colors[i], linewidth=0, label=f'sca, r@{1}')
    i += 1
    for ic in ['img', 'cap']:
        for r in [1, 5, 10]:
            if ic != 'img' or r != 1:
                ax2.plot(total_res_range, opt_acc[f'amb_{ic}ret_r{r}'], linestyle='solid', color=colors[i], linewidth=3.0, label=f'exh, r@{r}', alpha=0.4)
                ax2.plot(total_res_range, opt_acc[f'sca_{ic}ret_r{r}'], marker='*', markersize=15, color=colors[i], linewidth=0, label=f'sca, r@{r}', alpha=0.4)
                i += 1
    ax2.set_xlabel(r'$\bar{\mathbf{B}}$', fontsize=20)
    ax2.set_ylabel('Optimal accuracy', fontsize=15)
    ax2.legend(loc='upper left', fontsize='large')
    ax2.set_xticks(temp_res_oneside_range_db[1::3])
    ax2.set_xticklabels([f'2^{res_db}' for res_db in temp_res_oneside_range_db[1::3]])
    ax1.set_position([0.05, 0, 0.45, 1])
    ax2.set_position([0.55, 0.28, 0.4, 0.4])
    ax2.set_ylim(0, 100)
    plt.show()

