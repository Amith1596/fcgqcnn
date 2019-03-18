import sys
import os
import skimage
import matplotlib.pyplot as plt
import numpy as np

path= "/home/amithp/fcgqcnn_env/snapshots-thomas-converted/"
alt_path="/home/amithp/fcgqcnn_env/snapshots-thomas-noheight/"
folders=os.listdir(path)
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

#Uncomment for manually analysing certain images
# for folder in problem_images[:]:
#     depth_path=alt_path+folder+"/depth_0.npy"
#     color_path=alt_path+folder+"/color_0.png"
#     depth_path=path+folder+"/primesense/depth_0.npy"
#     color_path=path+folder+"/primesense/color_0.png"
    

#     color_img = plt.imread(color_path)
#     plt.subplot(121)
#     plt.imshow(color_img)

#     depth_img = np.load(depth_path)
#     plt.subplot(122)
#     plt.imshow(np.squeeze(depth_img))
#     plt.show()

#     os.system('python examples/policy.py FC-GQCNN-4.0-PJ --depth_image '+ depth_path + ' --id ' +folder+ ' --fully_conv')

for folder in folders[:]:
    img_path=alt_path+folder+"/depth_0.npy"
    #img_path=path+folder+"/primesense/depth_0.npy"
    
    #im = np.load(img_path)
    #plt.imshow(np.squeeze(im), cmap=plt.cm.gray_r)
    #plt.show()

    print 'python examples/policy.py FC-GQCNN-4.0-PJ --depth_image '+ img_path + ' --id ' +folder+ ' --fully_conv'
    os.system('python examples/policy.py FC-GQCNN-4.0-PJ --depth_image '+ img_path + ' --id ' +folder+ ' --fully_conv')
    