##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./imgs/undistort.png "Undistorted"
[image2]: ./imgs/ColorMaskCombined.png "Combined Color Mask"
[image3]: ./imgs/SobelLandS.png "Sobel on L and S channel"
[image4]: ./imgs/ColorandSobel.png "Color and Sobel Combined"
[image5]: ./imgs/warped.png "Warped Image"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

#### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines through 9 to 35 of the file called `helper.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The value of Camera Matrix and Calibration coeffient was stored into a pickle file.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.

First, I coverted the image to HSV color space. We are going to experiment in the HSV and HLS color space as its lightness component varies the most under different lightning condition but its Hue and Saturation channel stay fairly consistent in shadow or excessive brightness.This allows us to detect different color lane lines more realiabily than in any other color space.
The out from the yellow color mask and white color mask were combined together using bitwiseor operator to get the below output.
![alt text][image2]

Then, I converted the image to HLS space and applied Sobel filters on L and S channels. Lane lines tend to be close to be vertical. So we need to detect steep edges that are more likely to be a lane. Applying the Sobel operator to an image is a way of taking the derivative of the image in the x or y direction. Taking the gradient in the x direction emphasizes edges closer to vertical.  Alternatively, taking the gradient in the y direction emphasizes edges closer to horizontal.

helper.abs_sobel_thresh takes in an image and optional Sobel kernel size, as well as thresholds for gradient magnitude. 
![alt text][image3]

The above Sobel filter output was combined with color mask output to get the following output:
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 43 through 48 in the file `helper.py` (helper.py). The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner (See cell 4 `Final_AdvancedLaneFinding.pynb`(Final_AdvancedLaneFinding.pynb)):

```
img_size = np.shape(image)
#Select a trapezoid around the lane which needs to warped 
top_y = np.uint(img_size[0]/1.5)
bottom_y = np.uint(img_size[0])
center = np.uint(img_size[1]/2)
top_leftx = center - .2*np.uint(img_size[1]/2)
top_rightx = center + .2*np.uint(img_size[1]/2)
bottom_leftx = center - .9*np.uint(img_size[1]/2)
bottom_rightx = center + .9*np.uint(img_size[1]/2)

src = np.float32([[bottom_leftx,bottom_y],[bottom_rightx,bottom_y],[top_rightx,top_y],[top_leftx,top_y]])
dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],[img_size[1],0],[0,0]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

