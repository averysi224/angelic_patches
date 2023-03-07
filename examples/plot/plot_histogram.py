

# importing package
import matplotlib.pyplot as plt
import numpy as np
import seaborn
plt.style.use('seaborn-paper')
# plt.rcParams.update(plt.rcParamsDefault)

# aware + spatial
for opt in range(0):
	# create data
	x = np.arange(1,8)
	# Aware FRCNN
	if opt == 0:
		name = "frcnn1"
		y1 = [43.65, 36.51,25.39,17.68,8.52,4.10,18.30]
		y2 = [77.18, 78.23,60.14,35.94,47.16,61.24,32.60]
	elif opt == 1:
		name = "frcnn2"
		y1 = [83.08,86.54,71.83,62.65,58.58,78.55,71.71]
		y2 = [83.08,91.57,81.71,71.64,72.15,95.39,72.37]

	# # Aware SSD
	elif opt == 2:
		name = "ssd1"
		y1 = [37.82,25.69,4.70,12.10,1.72,10.63,7.32]
		y2 = [49.52,50.65,17.45,27.50,6.37,23.26,28.92]
	elif opt == 3:
		name = "ssd2"
		y1 = [64.85,60.49,23.90,37.33,23.02,63.04,40.14]
		y2 = [65.40,61.15,28.93,40.11,24.75,63.04,57.48]

	# Skew Aware FRCNN
	elif opt == 4:
		name = "affine_frcnn1"
		y1 = [18.21,5.32,5.06,2.38,0.88,0.35,3.60]
		y2 = [34.21,15.14,13.44,7.27,6.10,11.37,4.85]
	elif opt == 5:
		name = "affine_frcnn2"
		y1 = [47.31,20.50,18.73,15.44,10.53,11.95,21.84]
		y2 = [56.48,32.64,26.80,15.79,16.12,15.94,21.48]

	# # Skew Aware SSD
	elif opt == 6:
		name = "affine_ssd1"
		y1 = [22.80,5.82,0.88,5.61,0.56,0.80,1.90]
		y2 = [32.74,11.48,6.98,12.29,1.46,2.79,4.29]
	elif opt == 7:
		name = "affine_ssd2"
		y1 = [53.85,38.20,12.43,26.72,8.52,35.07,20.82]
		y2 = [54.00,41.67,22.18,30.21,8.69,35.50,27.55]

	width = 0.45
	plt.figure(figsize=(6,3))


	# plot data in grouped manner of bar type
	plt.bar(x, y1, width, color='#CC5A49') #'#7fc97f')
	plt.bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
	# plt.bar(x+0.4, y3, width, color='#386cb0')
	# plt.bar(x+0.8, y4, width, color='#f0027f')

	# plt.xticks(x, ['Bus', 'Cup', 'Person', 'Bottle', 'Bowl', 'Laptop', 'Chair'], fontsize=13)
	plt.xticks(x, ['Person', 'Bus', 'Bottle', 'Bowl', 'Chair', 'Laptop', 'Cup'], fontsize=13)
	plt.xlabel("Categories", fontsize=13)
	plt.ylabel("Recall (%)", fontsize=13)

	if opt % 2 == 0:
		plt.legend(["Corrupt, NoPatch", "Corrupt, Patch"], loc='upper right', fontsize=12)
	else:
		plt.legend(["Clear, NoPatch", "Clear, Patch"], loc='upper right', fontsize=12)

	# plt.legend(["Pert-NoPatch", "Pert-Patch", "No Patch", "Patch"], loc='upper right')
	# plt.show()
	plt.subplots_adjust(bottom=0.25)
	plt.savefig("results_"+name+"_auto")

# cross model
for opt in range(0):
	# create data
	x = np.arange(1,8)
	# Aware FRCNN
	if opt == 0:
		name = "yolo1"
		y1 = [51.70,42.00,27.01,21.40,3.24,3.56,28.87]
		y2 = [61.30,59.60,44.89,16.84,22.01,30.67,29.76]
	elif opt == 1:
		name = "yolo2"
		y1 = [75.17,84.55,56.98,60.34,45.18,75.44,60.25]
		y2 = [74.08,85.84,64.73,53.45,57.14,88.16,60.28]

	# # Aware SSD
	elif opt == 2:
		name = "focs1"
		y1 = [40.10,43.44,18.06,19.45,4.07,4.12,7.35]
		y2 = [58.15,69.68,45.49,20.14,20.99,28.81,27.94]
	elif opt == 3:
		name = "focs2"
		y1 = [75.36,80.53,63.29,58.88,54.13,70.35,65.83]
		y2 = [75.57,87.17,70.98,66.45,58.10,80.97,64.95]

	# Skew Aware FRCNN
	elif opt == 4:
		name = "reta1"
		y1 = [34.25,24.24,9.13,2.30,1.53,3.51,5.05]
		y2 = [54.74,49.35,27.76,2.68,11.62,18.86,9.09]
	elif opt == 5:
		name = "reta2"
		y1 = [72.99,67.86,46.06,25.18,22.96,51.95,46.53]
		y2 = [74.71,79.76,53.15,31.56,43.71,62.34,43.89]

	# Skew Aware FRCNN
	elif opt == 6:
		name = "frcnn"
		y1 = [44.57,36.15,25.69,16.78,7.27,4.33,18.30]
		y2 = [61.96,66.15,51.39,20.38,41.21,43.81,21.57]
	elif opt == 7:
		name = "ssd"
		y1 = [34.58,24.73,4.42,8.90,3.39,9.94,8.08]
		y2 = [43.61,41.21,13.50,20.61,14.12,21.74,13.06]

	width = 0.45
	plt.figure(figsize=(6,3))


	# plot data in grouped manner of bar type
	plt.bar(x, y1, width, color='#CC5A49') #'#7fc97f')
	plt.bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
	# plt.bar(x+0.4, y3, width, color='#386cb0')
	# plt.bar(x+0.8, y4, width, color='#f0027f')

	# plt.xticks(x, ['Bus', 'Cup', 'Person', 'Bottle', 'Bowl', 'Laptop', 'Chair'], fontsize=13)
	plt.xticks(x, ['Person', 'Bus', 'Bottle', 'Bowl', 'Chair', 'Laptop', 'Cup'], fontsize=13)
	plt.xlabel("Categories", fontsize=13)
	plt.ylabel("Recall (%)", fontsize=13)

	if opt % 2 == 0 or opt > 6:
		plt.legend(["Corrupt, NoPatch", "Corrupt, Patch"], loc='upper right', fontsize=12)
	else:
		plt.legend(["Clear, NoPatch", "Clear, Patch"], loc='upper right', fontsize=12)

	# plt.legend(["Pert-NoPatch", "Pert-Patch", "No Patch", "Patch"], loc='upper right')
	# plt.show()
	plt.subplots_adjust(bottom=0.25)
	plt.savefig("cross_"+name+"_auto")

# extra exp
for opt in range(6):
	# create data
	x = np.arange(1,8)
	# Aware FRCNN
	if opt == 0:
		name = "rand_frcnn"
		y1 = [48.18388195,38.35274542,24.30107527,18.92635926,7.435412728,6.818181818,23.77572747]
		y2 = [65.09648127,59.90016639,46.73835125,30.28217481,34.089477,45.8041958,24.9112846]
	elif opt == 1:
		name = "rand_ssd"
		y1 = [45.87155963,30.61389338,6.238185255,12.89707751,2.005899705,8.474576271,7.317073171]
		y2 = [49.31192661,42.24555735,10.39697543,17.47141042,4.719764012,18.72881356,28.91986063]

	# # Aware SSD
	elif opt == 2:
		name = "partial_frcnn1"
		y1 = [47.20,38.80,27.58,17.69,5.58,7.14,26.18]
		y2 = [63.29,60.95,51.69,27.91,28.97,53.35,32.62]
	elif opt == 3:
		name = "partial_ssd1"
		y1 = [38.67,29.93,5.42,12.01,3.22,9.41,8.18]
		y2 = [46.29,45.19,12.04,17.60,5.93,18.38,19.63]

	# Skew Aware FRCNN
	elif opt == 4:
		name = "partial_frcnn2"
		y1 = [87.15203426,86.46362098,71.00371747,60.25641026,60.74481074,79.17783735,73.48643006]
		y2 = [86.45610278,90.94754653,78.43866171,69.90553306,68.07081807,94.81680071,74.39109255]
	elif opt == 5:
		name = "partial_ssd2"
		y1 = [61.10,62.60,24.03,42.15,24.76,61.46,36.62]
		y2 = [61.86,66.19,29.83,44.65,25.50,59.51,47.34]

	width = 0.45
	plt.figure(figsize=(6,3))


	# plot data in grouped manner of bar type
	plt.bar(x, y1, width, color='#CC5A49') #'#7fc97f')
	plt.bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
	# plt.bar(x+0.4, y3, width, color='#386cb0')
	# plt.bar(x+0.8, y4, width, color='#f0027f')

	# plt.xticks(x, ['Bus', 'Cup', 'Person', 'Bottle', 'Bowl', 'Laptop', 'Chair'], fontsize=13)
	plt.xticks(x, ['Person', 'Bus', 'Bottle', 'Bowl', 'Chair', 'Laptop', 'Cup'], fontsize=13)
	plt.xlabel("Categories", fontsize=13)
	plt.ylabel("Recall (%)", fontsize=13)

	if opt < 4:
		plt.legend(["Corrupt, NoPatch", "Corrupt, Patch"], loc='upper right', fontsize=12)
	else:
		plt.legend(["Clear, NoPatch", "Clear, Patch"], loc='upper right', fontsize=12)

	# plt.legend(["Pert-NoPatch", "Pert-Patch", "No Patch", "Patch"], loc='upper right')
	# plt.show()
	plt.subplots_adjust(bottom=0.25)
	plt.savefig("extra_"+name+"_auto")
