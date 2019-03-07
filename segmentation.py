import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
#import pcl
import skimage
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage, SegmentationImage

path= "/home/amithp/fcgqcnn_env/snapshots-thomas-converted/"

folders=os.listdir(path)

#TRIAL THRESHOLDING METHODS

kernel = np.ones((4,4),np.uint8)
for folder in folders[:2]:
    exact_path=path+folder+"/primesense/color_0.png"

    img = cv2.imread(exact_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    print img.shape

    ret, th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th2= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, th4 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((4,4),np.uint8)
    erosion = cv2.erode(opening,kernel,iterations = 1)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 9)
    
    depth_data=np.load(path+folder+"/primesense/depth_0.npy")
    depth_data[depth_data > 0.675] = 0
    depth_data[depth_data != 0] = 1

    print depth_data


    titles = ['Original Image', 'OTSU',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [depth_data, opening, erosion, dilation]
    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    #plt.imshow(thresh,cmap=plt.cm.gray_r)
    plt.show()
    #cv2.imwrite(path+folder+"/primesense/segmask_0.png",thresh)



    #USING SIMILAR SEGMASK CREATION
    # camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','data/calib/primesense.intr')   
    # camera_intr = CameraIntrinsics.load(camera_intr_filename)

    #depth_data=np.load(path+folder+"/primesense/depth_0.npy")
    #depth_im = DepthImage(depth_data, frame=camera_intr.frame)

    # mask = np.zeros((camera_intr.height, camera_intr.width, 1), dtype=np.uint8)
    # c = np.array([165, 460, 500, 135])
    # r = np.array([165, 165, 370, 370])
    # rr, cc = skimage.draw.polygon(r, c, shape=mask.shape)
    # mask[rr, cc, 0] = 255
    # segmask = BinaryImage(mask)

    # valid_px_mask = depth_im.invalid_pixel_mask().inverse()

    # if segmask is None:
    #     segmask = valid_px_mask
    # else:
    #     segmask = segmask.mask_binary(valid_px_mask)

    # # create new cloud
    # point_cloud = camera_intr.deproject(depth_im)
    # point_cloud.remove_zero_points()
    # pcl_cloud = pcl.PointCloud(point_cloud.data.T.astype(np.float32))
    # tree = pcl_cloud.make_kdtree()

    # # find large clusters (likely to be real objects instead of noise)
    # ec = pcl_cloud.make_EuclideanClusterExtraction()
    # ec.set_ClusterTolerance(CLUSTER_TOL)
    # ec.set_MinClusterSize(MIN_CLUSTER_SIZE)
    # ec.set_MaxClusterSize(MAX_CLUSTER_SIZE)
    # ec.set_SearchMethod(tree)
    # cluster_indices = ec.Extract()
    # num_clusters = len(cluster_indices)

    # obj_segmask_data = np.zeros(depth_im.shape)
                        
    # # read out all points in large clusters
    # cur_i = 0
    # for j, indices in enumerate(cluster_indices):
    #     num_points = len(indices)
    #     points = np.zeros([3,num_points])
        
    #     for i, index in enumerate(indices):
    #         points[0,i] = pcl_cloud[index][0]
    #         points[1,i] = pcl_cloud[index][1]
    #         points[2,i] = pcl_cloud[index][2]
            
    #     segment = PointCloud(points, frame=camera_intr.frame)
    #     depth_segment = camera_intr.project_to_image(segment)
    #     obj_segmask_data[depth_segment.data > 0] = j+1
    # obj_segmask = SegmentationImage(obj_segmask_data.astype(np.uint8))
    # obj_segmask = obj_segmask.mask_binary(segmask)
    # plt.imshow(obj_segmask)
    # plt.show()