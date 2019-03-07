import numpy as np
import matplotlib.pyplot as plt 
from visualization import Visualizer2D as vis

preds=np.load('/home/amithp/fcgqcnn_env/output/trial.npy')
depth_im=np.load('/home/amithp/fcgqcnn_env/output/trial_depth.npy')

depth_im =depth_im[48:depth_im.shape[0] - 48, 48:depth_im.shape[1] - 48, 0]
depth_im = depth_im[::4, ::4]
print depth_im.shape
plt.imshow(depth_im,cmap=plt.cm.gray_r)
#plt.show()

print "shape of preds: "+ str(preds.shape)
#print preds
affordance_map=preds[:,:,:,:]
affordance_map=affordance_map.max(axis=3)
print affordance_map.shape
affordance_map=affordance_map.max(axis=0)
print affordance_map.shape
plt.imshow(affordance_map, cmap=plt.cm.RdYlGn, alpha=0.5, vmin=0.0, vmax=1.0)
affordance_argmax = np.unravel_index(np.argmax(affordance_map), affordance_map.shape)
print affordance_argmax
plt.scatter(affordance_argmax[1], affordance_argmax[0], c='black', marker='.', s=15)
plt.show()