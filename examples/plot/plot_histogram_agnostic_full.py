

# importing package
import matplotlib.pyplot as plt
import numpy as np
import pdb
import seaborn
plt.style.use('seaborn-paper')

# Some example data to display
x = np.arange(1,3)
y = np.sin(x ** 2)

fig, axs = plt.subplots(3, 5)
# Aware FRCNN
y1 = [34.4,46.4]  # [frcnn yx, ssd yx]
y2 = [43, 59.2]	# [frcnn yxx, ssd yxx]
width = 0.45
# axs[0, 0].plot(x, y)
axs[0, 0].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[0, 0].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[0, 0].set_title('Frost')
y1 = [51.6,66.1]  # [frcnn yx, ssd yx]
y2 = [59.3, 70.9]	# [frcnn yxx, ssd yxx]
axs[0, 1].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[0, 1].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[0, 1].set_title('Contrast')
y1 = [53.1, 60.9]  # [frcnn yx, ssd yx]
y2 = [63.2, 70.9]	# [frcnn yxx, ssd yxx]
axs[0, 2].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[0, 2].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[0, 2].set_title('Fog')
y1 = [64.8, 64.3]  # [frcnn yx, ssd yx]
y2 = [71.2, 79.5]	# [frcnn yxx, ssd yxx]
axs[0, 3].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[0, 3].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[0, 3].set_title('Brightness')
y1 = [73.3, 37.6]  # [frcnn yx, ssd yx]
y2 = [77.78, 45.3]	# [frcnn yxx, ssd yxx]
axs[0, 4].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[0, 4].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[0, 4].set_title('Gaussian Noise')
y1 = [62.2, 42.7]  # [frcnn yx, ssd yx]
y2 = [76, 55.6]	# [frcnn yxx, ssd yxx]
axs[1, 0].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[1, 0].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[1, 0].set_title('Short Noise')
y1 = [18.89, 29.1]  # [frcnn yx, ssd yx]
y2 = [26, 35.9]	# [frcnn yxx, ssd yxx]
axs[1, 1].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[1, 1].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[1, 1].set_title('Impulse Noise')
y1 = [74.4, 55.6]  # [frcnn yx, ssd yx]
y2 = [84.4, 56.9]	# [frcnn yxx, ssd yxx]
axs[1, 2].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[1, 2].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[1, 2].set_title('Speckle Noise')
y1 = [62.6, 45.6]  # [frcnn yx, ssd yx]
y2 = [71.1, 48.1]	# [frcnn yxx, ssd yxx]
axs[1, 3].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[1, 3].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[1, 3].set_title('Defocus Noise')
y1 = [65.9, 43.2]  # [frcnn yx, ssd yx]
y2 = [76.9, 49.2]	# [frcnn yxx, ssd yxx]
axs[1, 4].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[1, 4].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[1, 4].set_title('Motion Blur')
y1 = [57.9, 33.2]  # [frcnn yx, ssd yx]
y2 = [57.9, 39.2]	# [frcnn yxx, ssd yxx]
axs[2, 0].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[2, 0].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[2, 0].set_title('Zoom Blur')
y1 = [82.5, 52.7]  # [frcnn yx, ssd yx]
y2 = [87.3, 58.9]	# [frcnn yxx, ssd yxx]
axs[2, 1].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[2, 1].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[2, 1].set_title('Snow')
y1 = [54.76, 39.5]  # [frcnn yx, ssd yx]
y2 = [76.2, 39.5]	# [frcnn yxx, ssd yxx]
axs[2, 2].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[2, 2].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[2, 2].set_title('Spatter')
y1 = [41.8, 49.5]  # [frcnn yx, ssd yx]
y2 = [63.7, 43.9]	# [frcnn yxx, ssd yxx]
axs[2, 3].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[2, 3].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[2, 3].set_title('Glass Blur')
y1 = [28.1, 47.8]  # [frcnn yx, ssd yx]
y2 = [32.3, 57.4]	# [frcnn yxx, ssd yxx]
axs[2, 4].bar(x, y1, width, color='#CC5A49') #'#7fc97f')
axs[2, 4].bar(x+0.4, y2, width, color='#4586AC') #'#fdc086')
axs[2, 4].set_title('Pixlate')

fd={'fontsize': 9}
# 'fontweight': rcParams['axes.titleweight'],
# 'verticalalignment': 'baseline',
# fd={'horizontalalignment': 'right'}

for ax in axs.flat:
    ax.set(xlabel='', ylabel="Recall (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([' Faster-RCNN', '       SSD'], rotation=10, fontdict=fd)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.99,
                    top=0.93,
                    wspace=0.2,
                    hspace=0.5)
plt.savefig("results_agnostic")

