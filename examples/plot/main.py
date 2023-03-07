import matplotlib.pyplot as plt
import numpy as np
import seaborn
plt.style.use('seaborn-paper')
# plt.rcParams.update(plt.rcParamsDefault)


# create data
x = np.arange(1,8)

# labels = ['Bus', ' Cup', ' Person', ' Bottle', ' Bowl', ' Laptop', ' Chair']
# Affine Results - frcnn
# y1 = [4.03, 3.91, 23.53, 4.15, 6.73, 0.37, 0.30] #pert_no_patch
# y2 = [10.26, 8.54, 55.88, 13.84, 13.46, 7.01, 8.46] #pert_patch
# y1 = [18.68, 20.07, 38.61, 17.05, 16.30, 11.16, 14.53] #no_patch
# y2 = [28.02, 32.17, 54.19, 24.92, 26.33, 25.75, 24.02] #patch

# Affine Results - ssd
# y1 = [6.99, 1.83, 27.31, 1.39, 3.67, 2.79, 0.21] #pert_no_patch
# y2 = [14.34, 3.98, 37.89, 4.88, 12.99, 4.78, 2.29] #pert_patch
# y1 = [43.32, 19.94, 50.46, 13.03, 29.59, 46.44, 6.79] #no_patch
# y2 = [43.32, 14.53, 52.29, 26.06, 31.73, 43.45, 7.01] #patch

labels = ['person', ' bus', ' bottle', ' cup', ' bowl', ' chair', ' laptop']
# cross model 1: frcnn + ssd -> yolov5
# y1 = [18.02, 8.75, 6.09, 4.47, 5.97, 2.19, 2.41]
# y2 = [34.20, 18.21, 26.09, 8.88, 7.23, 4.64, 21.01]

y1 = [96.32,95.83,87.50,77.04,86.36,100.00, 85.71]
y2 = [95.52,95.99,96.77,89.87,88.46,94.44,98.12]

# cross model 2: frcnn + ssd -> focs
# y1 = [25.22, 28.36, 14.99, 14.62, 8.51, 5.80, 5.28]
# y2 = [40.87, 50.00, 29.39, 19.60, 17.63, 20.99, 29.27]
# y1 = [95.87,96.20,83.87,67.69,59.57,80.77,76.47]
# y2 = [97.92,94.37,87.93,83.10,82.86,89.41,90.00]

# # cross model 3: frcnn + ssd -> retina
# y1 = [34.29, 19.05, 13.42, 6.38, 2.71, 0.84, 0.80]
# y2 = [54.03, 42.86, 28.75, 8.05, 4.41, 8.40, 16.87]
# y1 = [94.96,97.96,93.33,76.00,88.89,75.00,66.67]
# y2 = [98.11,97.30,94.74,80.00,92.86,93.75,89.36]

width = 0.45
plt.figure(figsize=(6,3))

# plot data in grouped manner of bar type
plt.bar(x, y1, width, color='#CC5A49') #'#7fc97f')
plt.bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')

plt.xticks(x, labels, fontsize=13)
plt.xlabel("Categories", fontsize=13)
plt.ylabel("Precision (%)", fontsize=13)
plt.legend(["Corrupt, NoPatch", "Corrupt, Patch"], loc='upper right', fontsize=12)
# plt.legend(["Clear, NoPatch", "Clear, Patch"], loc='upper right', fontsize=12)

# plt.legend(["Pert-NoPatch", "Pert-Patch", "No Patch", "Patch"], loc='upper right')
# plt.show()
plt.subplots_adjust(bottom=0.25)
plt.savefig("cross_yolo5_precision")

# labels = ['gaussian noise', ' shot noise', ' impulse noise', ' speckle noise', ' gaussian blur', ' glass blur', ' defocus blur', ' motion blur', ' zoom blur', ' snow', ' spatter', ' saturate', ' jpeg_compression', ' pixelate']
# # person agnostic frcnn
# no_patch = [73.33, 62.22, 18.89, 74.44, 71.43, 41.76, 62.63, 65.93, 57.94, 82.54, 54.76, 87.30, 54.17, 28.13]
# patch = [77.78, 76, 21.11, 84.40, 71.43, 63.74, 71.11, 76.90, 57.94, 87.30, 76.20, 87.30, 60.42, 32.29]

# # person agnostic ssd
# no_patch = [37.61, 42.74, 29.10, 55.56, 57.80, 49.54, 46.42, 43.22, 33.33, 52.68, 39.47, 52.35, 54.55, 47.79]
# patch = [45.30, 55.56, 35.90, 55.56, 56.88, 33.94, 47.14, 49.15, 37.82, 58.93, 39.47, 51.35, 56.57, 57.35]


