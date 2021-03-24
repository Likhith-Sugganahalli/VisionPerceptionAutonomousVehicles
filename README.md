  VisionPerceptionAutonomousVehicles
code i wrote for Self-Driving Vehicles Course on Coursera 


  code to estimate an autonomous vehicle trajectory by images taken with a monocular camera set up on the vehicle.
  
  
  Steps taken
  - Extract  features from the photographs  taken with a camera setup on the vehicle.
  - Use the extracted features to find matches between the features in different photographs.
  - Use the found matches to estimate the camera motion between subsequent photographs. 
  - Use the estimated camera motion to build the vehicle trajectory.


  The dataset handler contains 52 data frames. Each frame contains an RGB image and a depth map taken with a setup on the vehicle and a grayscale version of the RGB image which will be used for computation. Furthermore, camera calibration matrix K is also provided in the dataset handler.
  
  Upon creation of the dataset handler object, all the frames will be automatically read and loaded. The frame content can be accessed by using `images`, `images_rgb`, `depth_maps` attributes of the dataset handler object along with the index of the requested frame. 

  **Note (Depth Maps)**: Maximum depth distance is 1000. This value of depth shows that the selected pixel is at least 1000m (1km) far from the camera, however the exact distance of this pixel from the camera is unknown. Having this kind of points in further trajectory estimation might affect the trajectory precision.

 
  Extracting Features from an Image
  
  i use ORB feature descriptor  
  
  
  Reference to https://docs.opencv.org/3.4.3/d2/d29/classcv_1_1KeyPoint.html
  .
  
  Also extracted features with different descriptors such as SIFT, ORB, SURF and BRIEF, tried using detectors such as Harris corners or FAST and pairing them with a descriptor. L

  Reference : https://docs.opencv.org/3.4.3/db/d27/tutorial_py_table_of_contents_feature2d.html


  
  Trajectory Estimation
  
  At this point I have everything to perform visual odometry for the autonomous vehicle. now i incrementally estimate the pose of the vehicle by examining the changes that motion induces on the images of its onboard camera.
  
  

  Estimating Camera Motion between a Pair of Images
  
  Implemented camera motion estimation from a pair of images 

  Used motion estimation algorithm, namely Perspective-n-Point (PnP), as well as Essential Matrix Decomposition.
  
  To use PnP, depth maps of frame are needed, C


  To use Essential Matrix Decomposition, reference : https://en.wikipedia.org/wiki/Essential_matrix
  
  More information on both approaches implementation can be found in https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html . Specifically, Detailed Description_ section of [https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html , as it explains the connection between the 3D world coordinate system and the 2D image coordinate system.
  
  


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
               


    
    
  Camera Movement Visualization

  visualize_camera_movement function description:
  

  Arguments:
  image1 -- the first image in a matched image pair (RGB or grayscale)
  image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [x, y], where x and y are 
                   coordinates of the i-th match in the image coordinate system
  image2 -- the second image in a matched image pair (RGB or grayscale)
  image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [x, y], where x and y are 
                   coordinates of the i-th match in the image coordinate system
  is_show_img_after_mov -- a boolean variable, controling the output (read image_move description for more info) 
  
  Returns:
  image_move -- an image with the visualization. When is_show_img_after_mov=False then the image points from both images are visualized on the first image. Otherwise, the image points from the second image only are visualized on the second image
  

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
