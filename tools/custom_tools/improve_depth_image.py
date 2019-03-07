import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import os

path='/home/amithp/fcgqcnn_env/snapshots-thomas-noheight/'
folders = os.listdir(path)
threshold=5
def search(color_val,depth_im,color_im,index):
    i=0
    while i <= 5:
        print i
        try:
            if(color_im[index[0]+i,index[1]]<color_val+threshold and color_im[index[0]+i,index[1]]>color_val-threshold and depth_im[index[0]+i,index[1]]!=0):
                return depth_im[index[0]+i,index[1]]
            if(color_im[index[0]-i,index[1]]<color_val+threshold and color_im[index[0]-i,index[1]]>color_val-threshold and depth_im[index[0]-i,index[1]]!=0):
                return depth_im[index[0]-i,index[1]]
            if(color_im[index[0],index[1]+i]<color_val+threshold and color_im[index[0],index[1]+i]>color_val-threshold and depth_im[index[0],index[1]+i]!=0):
                return depth_im[index[0],index[1]+i]
            if(color_im[index[0],index[1]-i]<color_val+threshold and color_im[index[0],index[1]-i]>color_val-threshold and depth_im[index[0],index[1]-i]!=0):
                return depth_im[index[0],index[1]-i]
        except:
            print "out of bound issue"
        i=i+1
    print "not found"  
    return 0

for folder in folders[:1]:
    # depth_im=np.load(path+folder+'/depth_0.npy')
    # color_im=cv2.imread(path+folder+'/color_0.png')
    depth_im=np.load(path+'2019-02-13-18-53-3512345'+'/depth_0.npy')
    color_im=cv2.imread(path+'2019-02-13-18-53-3512345'+'/color_0.png')
    color_im=cv2.cvtColor(color_im,cv2.COLOR_BGR2RGB)
    color_im_gray=cv2.cvtColor(color_im,cv2.COLOR_RGB2GRAY)
    plt.subplot(221)
    plt.imshow(color_im)
    plt.subplot(222)
    plt.imshow(depth_im,cmap=plt.cm.gray_r)
    plt.subplot(223)
    plt.imshow(color_im_gray,cmap=plt.cm.gray)
    check=1
    depth_im_data = depth_im[:, :, np.newaxis]
    zero_px = np.where(np.sum(depth_im_data, axis=2) == 0)
    zero_px = np.c_[zero_px[0], zero_px[1]]
    print zero_px
    print zero_px.shape
    while(check<5):
        for pixel in zero_px:
            rgb_val=color_im_gray[pixel[0],pixel[1]]
            depth_fill=search(rgb_val,depth_im,color_im_gray,pixel)
            if(depth_fill!=0):
                print "found "+str(depth_fill)
            else:
                print "not found"
            depth_im[pixel[0],pixel[1]]=depth_fill
        depth_im_data = depth_im[:, :, np.newaxis]
        zero_px = np.where(np.sum(depth_im_data, axis=2) == 0)
        zero_px = np.c_[zero_px[0], zero_px[1]]
        print zero_px.shape
        check=check+1
    
    plt.subplot(224)
    plt.imshow(depth_im,cmap=plt.cm.gray_r)
    
    
    plt.show()

    print type(color_im_gray)
    print color_im_gray.shape
