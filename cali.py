import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob



def calib_pixel():
    checkerboard=(7, 7)
    square_length=22,
    image_folder_path = 'calibration_image'
    sample_image_path = None
    # Prepare object points based on the real-world dimensions of the checkerboard
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[1:checkerboard[0]+1, 1:checkerboard[1]+1].T.reshape(-1, 2) * square_length

    # Prepare arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []

    # Load all imagesa
    images = glob.glob(image_folder_path + '/*.jpg')

    # Iterate through each image
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                       (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1))
            imgpoints.append(corners2)

            

    # Perform camera calibration to find the camera matrix, distortion coefficients, etc.
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the calibration results
    print(f"Reprojection error: {ret}")
    print(f"Camera matrix (K): \n{K}")
    print(f"Distortion coefficients (dist): \n{dist}")

    if sample_image_path is None:
        # Use the first image for undistortion if no sample image path is provided
        sample_image_path = images[0]

    # Load a sample image and undistort it using the calibration parameters
    img = cv2.imread(sample_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Undistort the image
    h, w = gray.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, K, dist, None, new_camera_matrix)

    # Find the chessboard corners in the undistorted image
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, 
                                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1))
        
        # Choose two points from the detected corners (e.g., (0,0) and (1,0) corner of the first square)
        p1 = corners2[0][0]  # Top-left corner of the first square
        p2 = corners2[1][0]  # Top-right corner of the first square

        # Compute the distance between the points in pixels
        pixel_distance = np.linalg.norm(p2 - p1)
        
                # Compute the distance between the points in pixels
        pixel_distance = np.linalg.norm(p2 - p1)
        
        # Create a blank image to draw the points and lines
        #blank_img = np.zeros_like(undistorted_img)
        
        # Draw the points and the line between them on the blank image
        #cv2.circle(blank_img, tuple(p1.ravel().astype(int)), 10, (0, 0, 255), -1)  # Red circle at p1
        #cv2.circle(blank_img, tuple(p2.ravel().astype(int)), 10, (0, 255, 0), -1)  # Green circle at p2
        #cv2.line(blank_img, tuple(p1.ravel().astype(int)), tuple(p2.ravel().astype(int)), (255, 0, 0), 2)  # Blue line between the points

        # Overlay the blank image with markings onto the undistorted image
        #combined_img = cv2.addWeighted(undistorted_img, 0.7, blank_img, 0.3, 0)

        # Display the image with the two points marked
        #plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        #plt.title(f'Distance in pixels: {pixel_distance:.2f}')
        #plt.show()

        print(f"Distance between points in pixels: {pixel_distance:.2f}")
        
        return pixel_distance
    else:
        print("Chessboard corners not found in the undistorted image.")
        return None



