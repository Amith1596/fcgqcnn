import csv
import numpy as np
import matplotlib.pyplot as plt

q = []
with open('/home/amithp/fcgqcnn_env/out_filter.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] != '':
            q.append(float(row[2]))
q_mean = np.mean(q)

q_fixed = []
with open('/home/amithp/fcgqcnn_env/out.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] !='':
            q_fixed.append(float(row[2]))
q_fixed_mean = np.mean(q_fixed)

q_threshold=[]
with open('/home/amithp/fcgqcnn_env/out_threshold_segmask.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] !='':
            q_threshold.append(float(row[2]))
q_threshold_mean = np.mean(q_threshold)

q_noheight=[]
with open('/home/amithp/fcgqcnn_env/out_threshold_no_height.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[2] !='':
            q_noheight.append(float(row[2]))
q_noheight_mean = np.mean(q_noheight)


ax1=plt.subplot(221)
n1, _, _ = ax1.hist(q, range=(0, 1.0), bins=100)
ax1.set_title('Threshold Segmask + Filter')

ax2=plt.subplot(222)
n2, _, _ = ax2.hist(q_fixed, range=(0, 1.0), bins=100)
ax2.set_title('Fixed Segmask')

ax3=plt.subplot(223)
n3, _, _ = ax3.hist(q_threshold, range=(0, 1.0), bins=100)
ax3.set_title('Threshold Segmask')

ax4=plt.subplot(224)
n4, _, _ = ax4.hist(q_noheight, range=(0, 1.0), bins=100)
ax4.set_title('Threshold Segmask (table height not adjusted)')

# Set y limit
ymax = np.max([n1.max(), n2.max(),n3.max(),n4.max()]) + 0.5
ax1.set_ylim((0, ymax))
ax2.set_ylim((0, ymax))
ax3.set_ylim((0, ymax))
ax4.set_ylim((0, ymax))

# Show mean
ax1.vlines([q_mean], 0, ymax, colors='r')
ax2.vlines([q_fixed_mean], 0, ymax, colors='r')
ax3.vlines([q_threshold_mean], 0, ymax, colors='r')
ax4.vlines([q_noheight_mean], 0, ymax, colors='r')

plt.show()