#!/usr/bin/env python3
# coding: utf-8

# # Visual Odometry for Localization in Autonomous Driving
# 
# Welcome to the assignment for Module 2: Visual Features - Detection, Description and Matching. In this assignment, you will practice using the material you have learned to estimate an autonomous vehicle trajectory by images taken with a monocular camera set up on the vehicle.
# 
# 
# **In this assignment, you will:**
# - Extract  features from the photographs  taken with a camera setup on the vehicle.
# - Use the extracted features to find matches between the features in different photographs.
# - Use the found matches to estimate the camera motion between subsequent photographs. 
# - Use the estimated camera motion to build the vehicle trajectory.
# 
# For most exercises, you are provided with a suggested outline. You are encouraged to diverge from the outline if you think there is a better, more efficient way to solve a problem.
# 
# You are only allowed to use the packages loaded bellow and the custom functions explained in the notebook. Run the cell bellow to import the required packages:

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)



# ## 0 - Loading and Visualizing the Data
# We provide you with a convenient dataset handler class to read and iterate through samples taken from the CARLA simulator. Run the following code to create a dataset handler object. 

# In[2]:


dataset_handler = DatasetHandler()


# The dataset handler contains 52 data frames. Each frame contains an RGB image and a depth map taken with a setup on the vehicle and a grayscale version of the RGB image which will be used for computation. Furthermore, camera calibration matrix K is also provided in the dataset handler.
# 
# Upon creation of the dataset handler object, all the frames will be automatically read and loaded. The frame content can be accessed by using `images`, `images_rgb`, `depth_maps` attributes of the dataset handler object along with the index of the requested frame. See how to access the images (grayscale), rgb images (3-channel color), depth maps and camera calibration matrix in the example below.
# 
# **Note (Depth Maps)**: Maximum depth distance is 1000. This value of depth shows that the selected pixel is at least 1000m (1km) far from the camera, however the exact distance of this pixel from the camera is unknown. Having this kind of points in further trajectory estimation might affect the trajectory precision.

# In[3]:


image = dataset_handler.images[0]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')


# In[4]:


image_rgb = dataset_handler.images_rgb[0]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image_rgb)


# In[5]:


i = 0
depth = dataset_handler.depth_maps[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(depth, cmap='jet')


# In[6]:


print("Depth map shape: {0}".format(depth.shape))

v, u = depth.shape
depth_val = depth[v-1, u-1]
print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(i, depth_val))


# In[7]:


dataset_handler.k


# In order to access an arbitrary frame use image index, as shown in the examples below. Make sure the indexes are within the number of frames in the dataset. The number of frames in the dataset can be accessed with num_frames attribute.

# In[8]:


# Number of frames in the dataset
print(dataset_handler.num_frames)


# In[9]:


i = 30
image = dataset_handler.images[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')


# ## 1 - Feature Extraction
# 
# ### 1.1 - Extracting Features from an Image
# 
# **Task**: Implement feature extraction from a single image. You can use any feature descriptor of your choice covered in the lectures, ORB for example. 
# 
# 
# Note 1: Make sure you understand the structure of the keypoint descriptor object, this will be very useful for your further tasks. You might find [OpenCV: Keypoint Class Description](https://docs.opencv.org/3.4.3/d2/d29/classcv_1_1KeyPoint.html) handy.
# 
# Note 2: Make sure you understand the image coordinate system, namely the origin location and axis directions.
# 
# Note 3: We provide you with a function to visualise the features detected. Run the last 2 cells in section 1.1 to view.
# 
# ***Optional***: Try to extract features with different descriptors such as SIFT, ORB, SURF and BRIEF. You can also try using detectors such as Harris corners or FAST and pairing them with a descriptor. Lastly, try changing parameters of the algorithms. Do you see the difference in various approaches?
# You might find this link useful:  [OpenCV:Feature Detection and Description](https://docs.opencv.org/3.4.3/db/d27/tutorial_py_table_of_contents_feature2d.html). 

# In[10]:


def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    ### START CODE HERE ### 
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)

    
    ### END CODE HERE ###
    
    return kp, des


# In[11]:


# In[12]:


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)



def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []
    
    ### START CODE HERE ###
    for i in images:
        kp, des = extract_features(i)
        kp_list.append(kp)
        des_list.append(des)

    
    ### END CODE HERE ###
    
    return kp_list, des_list


# In[15]:


images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    ### START CODE HERE ###
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    """
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    total_len = len(matches)
    to_remove = round(4*total_len/5)
    del matches[-to_remove:]
    """
    
    ### END CODE HERE ###

    return matches



def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    matches = sorted(match, key = lambda x:x.distance)
    ### START CODE HERE ###
    best = matches[0]
    
    for i in matches:
        if best.distance/i.distance < dist_threshold :
            filtered_match.append(i)

    return filtered_match



# In[187]:


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None,flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


# In[190]:


# Visualize n first matches, set n to None to view all matches
# set filtering to True if using match filtering, otherwise set to False
n = None
filtering = True

i = 0 
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
if filtering:
    dist_threshold = 0.2
    match = filter_matches_distance(match, dist_threshold)

image_matches = visualize_matches(image1, kp1, image2, kp2, match[:n])    


# ### 2.2 - Matching Features in Each Subsequent Image Pair in the Dataset
# 
# **Task**: Implement feature matching for each subsequent image pair in the dataset with the function you wrote in the above section.
# 
# ***Optional***: Implement match filtering by thresholding the distance for each subsequent image pair in the dataset with the function you wrote in the above section.

# In[50]:


def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = []
    
    ### START CODE HERE ###
    for i in range(len(des_list)-1):
        matches.append(match_features(des_list[i], des_list[i+1]))

        #print()

    
    ### END CODE HERE ###
    
    return matches


# In[114]:


matches = match_features_dataset(des_list, match_features)
#print(len(des_list))

#print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))

i=48
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]
image_matches = visualize_matches(image1, kp1, image2, kp2, matches[i][:n]) 
print(len(matches[i]))


# In[236]:


# Optional
def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = map(filter_matches_distance,matches)
    m
    filtered_match = filter_matches_distance(i, dist_threshold)
    filtered_matches.append(filtered_match)
    ### START CODE HERE ###


    
    ### END CODE HERE ###
    
    return filtered_matches


# In[237]:


# Optional
dist_threshold = 0.2

#filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
    is_main_filtered_m = False
    if is_main_filtered_m: 
        matches = filtered_matches

    i = 0
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))
    
#image_matches = visualize_matches(image1, kp1, image2, kp2, filtered_matches[:n])


# ## 3 - Trajectory Estimation
# 
# At this point you have everything to perform visual odometry for the autonomous vehicle. In this section you will incrementally estimate the pose of the vehicle by examining the changes that motion induces on the images of its onboard camera.
# 
# ### 3.1 - Estimating Camera Motion between a Pair of Images
# 
# **Task**: Implement camera motion estimation from a pair of images. You can use the motion estimation algorithm covered in the lecture materials, namely Perspective-n-Point (PnP), as well as Essential Matrix Decomposition.
# 
# - If you decide to use PnP, you will need depth maps of frame and they are provided with the dataset handler. Check out Section 0 of this assignment to recall how to access them if you need. As this method has been covered in the course, review the lecture materials if need be.
# - If you decide to use Essential Matrix Decomposition, more information about this method can be found in [Wikipedia: Determining R and t from E](https://en.wikipedia.org/wiki/Essential_matrix).
# 
# More information on both approaches implementation can be found in [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html). Specifically, you might be interested in _Detailed Description_ section of [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html) as it explains the connection between the 3D world coordinate system and the 2D image coordinate system.
# 
# 
# ***Optional***: Implement camera motion estimation with PnP, PnP with RANSAC and Essential Matrix Decomposition. Check out how filtering matches by distance changes estimated camera movement. Do you see the difference in various approaches?

# In[147]:


def estimate_motion(match, kp1, kp2, k, depth1):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = []
    image2_points = []
    objectpoints = np.array([[],[],[]],dtype="float64")
    #i = 0
    print(len(match))
    for mi in match:       
        x1, y1 = kp1[mi.queryIdx].pt
        
        y1 = int(y1)
        x1 = int(x1)
        Z = depth1[int(y1), int(x1)]
        
        if Z < 1000:
            image1_points.append([x1, y1])
            
            x2, y2 = kp2[mi.trainIdx].pt
            y2 = int(y2)
            x2 = int(x2)
            image2_points.append([x2, y2])
            
            scaled_coord = np.dot(np.linalg.inv(k), np.array([x1, y1, 1]).reshape([3,1]))
            cali_coord = scaled_coord * Z
            #import pdb; pdb.set_trace()
            objectpoints = np.c_[objectpoints, cali_coord]
            #i = i + 1
        else:
            continue
            
            
    objectpoints = objectpoints.T
    imagepoints = np.array(image2_points,dtype="float32")
    dist_coef = np.zeros(4)
    #objectpoints = objectpoints.T
    #imagepoints = np.array(image2_points)
    print(objectpoints.shape)
    print(imagepoints.shape)
    print(len(image1_points))
    print(len(image2_points))
    ret,rmat,tvec = cv2.solvePnP(objectpoints,imagepoints, k,dist_coef)
    #import pdb; pdb.set_trace()
        
    ### END CODE HERE ###
    
    return rmat, tvec, image1_points, image2_points


# In[148]:


i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[i]
#print(match)
#(x,y) = kp1.kp
##print(kp1.pt)
rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))


# **Expected Output Format**:
# 
# Make sure that your estimated rotation matrix and translation vector are in the same format as the given initial values
# 
# ```
# rmat = np.eye(3)
# tvec = np.zeros((3, 1))
# 
# print("Initial rotation:\n {0}".format(rmat))
# print("Initial translation:\n {0}".format(tvec))
# ```
# 
# 
# ```
# Initial rotation:
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
# Initial translation:
#  [[0.]
#  [0.]
#  [0.]]
# ```

# **Camera Movement Visualization**:
# You can use `visualize_camera_movement` that is provided to you. This function visualizes final image matches from an image pair connected with an arrow corresponding to direction of camera movement (when `is_show_img_after_mov = False`). The function description:
# ```
# Arguments:
# image1 -- the first image in a matched image pair (RGB or grayscale)
# image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [x, y], where x and y are 
#                  coordinates of the i-th match in the image coordinate system
# image2 -- the second image in a matched image pair (RGB or grayscale)
# image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [x, y], where x and y are 
#                  coordinates of the i-th match in the image coordinate system
# is_show_img_after_mov -- a boolean variable, controling the output (read image_move description for more info) 
# 
# Returns:
# image_move -- an image with the visualization. When is_show_img_after_mov=False then the image points from both images are visualized on the first image. Otherwise, the image points from the second image only are visualized on the second image
# ```

# In[149]:


i=48
image1  = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)


# In[150]:


image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
# These visualizations might be helpful for understanding the quality of image points selected for the camera motion estimation


# ### 3.2 - Camera Trajectory Estimation
# 
# **Task**: Implement camera trajectory estimation with visual odometry. More specifically, implement camera motion estimation for each subsequent image pair in the dataset with the function you wrote in the above section.
# 
# ***Note***: Do not forget that the image pairs are not independent one to each other. i-th and (i + 1)-th image pairs have an image in common

# In[169]:


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function

    """
    trajectory = [np.array([0, 0, 0])]
    P = np.eye(4)
    
    for i in range(len(matches)):
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth = depth_maps[i]

        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth)
        R = rmat
        t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
        
        P_new = np.eye(4)
        P_new[0:3,0:3] = R.T
        P_new[0:3,3] = (-R.T).dot(t)
        P = P.dot(P_new)

        trajectory.append(P[:3,3])

    trajectory = np.array(trajectory).T 
    trajectory[2,:] = -1*trajectory[2,:]
    
    return trajectory


# In[170]:


depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

# Remember that the length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))


# **Expected Output**:
# 
# ```
# Camera location in point i is: 
#  [[locXi]
#  [locYi]
#  [locZi]]```
#  
#  In this output: locXi, locYi, locZi are the coordinates of the corresponding i-th camera location

# ## 4 - Submission:
# 
# Evaluation of this assignment is based on the estimated trajectory from the output of the cell below.
# Please run the cell bellow, then copy its output to the provided yaml file for submission on the programming assignment page.
# 
# **Expected Submission Format**:
# 
# ```
# Trajectory X:
#  [[  0.          locX1        locX2        ...   ]]
# Trajectory Y:
#  [[  0.          locY1        locY2        ...   ]]
# Trajectory Z:
#  [[  0.          locZ1        locZ2        ...   ]]
# ```
#  
#  In this output: locX1, locY1, locZ1; locX2, locY2, locZ2; ... are the coordinates of the corresponding 1st, 2nd and etc. camera locations

# In[ ]:


# Note: Make sure to uncomment the below line if you modified the original data in any ways
#dataset_handler = DatasetHandler()


# Part 1. Features Extraction
images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)


# Part II. Feature Matching
matches = match_features_dataset(des_list, match_features)

# Set to True if you want to use filtered matches or False otherwise
is_main_filtered_m = True
if is_main_filtered_m:
    dist_threshold = 0.75
    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches

    
# Part III. Trajectory Estimation
depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)


#!!! Make sure you don't modify the output in any way
# Print Submission Info
print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))


# ### Visualize your Results
# 
# **Important**:
# 
# 1) Make sure your results visualization is appealing before submitting your results. You might want to download this project dataset and check whether the trajectory that you have estimated is consistent to the one that you see from the dataset frames. 
# 
# 2) Assure that your trajectory axis directions follow the ones in _Detailed Description_ section of [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html).

# In[171]:


visualize_trajectory(trajectory)


# Congrats on finishing this assignment! 
