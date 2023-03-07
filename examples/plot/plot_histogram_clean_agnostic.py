

# importing package
import matplotlib.pyplot as plt
import numpy as np

# create data
x = np.arange(1,8)
# # Aware FRCNN
y1 = [77.51,56.23,81.43,67.31,46.05,62.40,44.18]
y11 = [85.57,58.79,89.01,77.56,50.34,72.00,48.36]

# # Aware SSD
# y1 = [70.50,53.82,63.93,20.53,44.18,66.67,30.16]
# y11 = [81.23,56.27,65.57,25.81,48.36,72.87,34.80]

width = 0.3
plt.figure(figsize=(4,2.5))


# plot data in grouped manner of bar type
plt.bar(x, y11, width, color='#386cb0')
# plt.bar(x+0.13, y22, width, color='#fdc086')
# plt.bar(x+0.36, y33, width, color='#4d8de5')
# plt.bar(x+0.59, y44, width, color='#ff0084')

plt.bar(x-0.3, y1, width, color='#7fc97f')
# plt.bar(x+0.03, y2, width, color='#dda676')

# plt.bar(x+0.26, y3, width, color='#386cb0')
# plt.bar(x+0.49, y4, width, color='#f0027f')

plt.xticks(x, ['Bus', 'Cup', 'Person', 'Bottle', 'Bowl', 'Laptop', 'Chair'])
plt.xlabel("Categories")
plt.ylabel("Recall (%)")
plt.legend(["Clean", "Patch"], loc='upper right')
# plt.show()
plt.savefig("results_agnostic_clean")
