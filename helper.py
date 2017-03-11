import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import cv2
import pickle

#Function to to get Camera Matrix and Calibration coeffient
def cameraPropFinder(nx,ny,cal_img_location, confFileLocation):
	images = glob.glob(cal_img_location)
	objpoints = []
	imgpoints = []
	objp = np.zeros((ny*nx,3),np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

	for fname in images:
		img = mpimg.imread(fname)
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		
		# If found, draw corners
		if ret == True:
			# Draw and display the corners
			cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			objpoints.append(objp)
			imgpoints.append(corners)
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	data = [mtx,dist]
	# Save the camera matrix and calibration coeffient to a file
	pickle.dump( data, open( confFileLocation, "wb" ) )
	last_img = img; #Send back last image
	return last_img

#Function to undistort an image using Camera Matrix and Calibration coeffient
def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
	
#Function to perform perscpective transform on an image
def warp_image(img,src,dst):
	img_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	Minv = cv2.getPerspectiveTransform(dst, src)
	return warped,M,Minv

	
# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output
    
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.

# Run the function
#grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    img_sx = cv2.Sobel(img,cv2.CV_64F,1,0, ksize=sobel_kernel)
    img_sy = cv2.Sobel(img,cv2.CV_64F,0,1, ksize=sobel_kernel)
    grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))
    binary_output[(grad_s>=thresh[0]) & (grad_s<=thresh[1])] = 1
    return binary_output
	
# Apply color mask to image
def apply_color_mask(hsv,img,low,high):
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res
	
# Return mask from HSV 	
def color_mask(hsv,low,high):

    mask = cv2.inRange(hsv, low, high)
    return mask


# Moving average
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# draw lines
def draw_pw_lines(img,pts,color):
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i+1][0]
        y2 = pts[0][i+1][1]
        cv2.line(img, (x1, y1), (x2, y2),color,50)

# This function returns masks for points used in computing polynomial fit. 
def get_mask_poly(img,poly_fit,window_sz):
    mask_poly = np.zeros_like(img)
    img_size = np.shape(img)
    poly_pts = []
    pt_y_all = []
    for i in range(8):
        img_y1 = img_size[0]-img_size[0]*i/8
        img_y2 = img_size[0]-img_size[0]*(i+1)/8
        pt_y = (img_y1+img_y2)/2
        pt_y_all.append(pt_y)
        poly_pt = np.round(poly_fit[0]*pt_y**2 + poly_fit[1]*pt_y + poly_fit[2])
        poly_pts.append(poly_pt)
        mask_poly[img_y2:img_y1,poly_pt-window_sz:poly_pt+window_sz] = 1.     
    return mask_poly, np.array(poly_pts),np.array(pt_y_all)

# Returns value of a quadratic polynomial 
def get_val(y,pol_a):
    return pol_a[0]*y**2+pol_a[1]*y+pol_a[2]