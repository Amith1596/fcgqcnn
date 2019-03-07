import sys
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import stats

#Base paths
path= "/home/amithp/fcgqcnn_env/snapshots-thomas-converted/"
alt_path="/home/amithp/fcgqcnn_env/snapshots-thomas-noheight/"

problem_images=["2019-02-13-18-53-5412345",
"2019-02-13-19-15-2512345",
"2019-02-13-19-10-3012345",
"2019-02-13-19-43-3512345",
"2019-02-13-19-34-3612345",
"2019-02-13-19-00-3712345",
"2019-02-13-18-54-5912345",
"2019-02-13-18-54-0712345",
"2019-02-13-19-03-0712345"]

new_problem_images=["2019-02-13-19-35-5212345",
"2019-02-13-19-34-5312345",
"2019-02-13-19-34-3612345",
"2019-02-13-18-57-5412345",
"2019-02-13-19-46-2512345",
"2019-02-13-19-09-4412345",
"2019-02-13-19-16-5512345",
"2019-02-13-19-35-3612345",
"2019-02-13-19-05-0912345"]

# Parameters
input_dirname = 'snapshots-thomas' if len(sys.argv) < 2 else sys.argv[1]
output_base_dir = 'snapshots-thomas-noheight' if len(sys.argv) < 2 else sys.argv[2] 
sensor = 'primesense'

# Get files
flag=1
a=0
fnames = os.listdir(input_dirname)

#uncomment for analysing problem images

# for image in new_problem_images:
#     im = plt.imread("images/"+image+".png")
#     plt.title(image)
#     plt.imshow(im)
#     plt.show()


for fname in fnames[:]:
    if 'Depth' in fname:
        #print fname[:24]
        if flag == 1:# and fname[:24] not in problem_images:
            depth_im = skimage.data.imread("%s/%s"%(input_dirname, fname), as_gray=True)
            depth_im = (depth_im / 1000.0)
            #depth_im[depth_im > 1] = 2

            if depth_im.max() > 0.8:
                a+=1
                depth_im[depth_im > 0.8] = 2
                print fname[:24]+"has a max value of " +str(depth_im.max())
                color_im=plt.imread(output_base_dir+'/'+fname[:24]+'/color_0.png')
                plt.subplot(1,2,1)
                plt.title('Max depth '+str(depth_im.max()))
                plt.imshow(color_im)
                plt.subplot(1,2,2)
                plt.title(fname[:24])
                plt.imshow(depth_im, cmap=plt.cm.gray_r , norm=plt.Normalize(0, 2))
                #plt.savefig(fname[:24]+".png")
                plt.show()