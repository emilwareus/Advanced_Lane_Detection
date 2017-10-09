
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

[image1]: ./output_images/Distort.PNG "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/thresh_img.png "Binary Example"
[image4]: ./output_images/Car_w_lanes.PNG "Output"
[video1]: ./output_1.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Distort.py file and calibrate function. This function contains the reading of the chessboard images and the generation of "mtx" and "dist" for calibration. This function is called upon in the Lane_Finding_main.py file. Every frame is undistroted with these paramters.  


![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. In the file `Thresh.py` you can find the following methods: 
* def dir_threashold(img, sobel_kernel = 3, thresh = (0, np.pi/2))
* def mag_threashold(img, sobel_kernel = 3, thresh = (0, 255))
* def abs_sobel_thresh(img, sobel_kernel=3, orient = 'x', thresh = (0, 255))
* def color_thresh(img, thresh_s = (70, 185), thresh_h = (15, 30))

There were combined as: combined[(((gradx == 1) & (grady == 1)) |((mag == 1) & (dir_binary == 1)) | (color==1))] = 1

Here you can see a result from the the transformation turned out. 
A cv2.fillPoly(mask, vertices, ignore_mask_color) mask was applyed as well to reduce noise in the image. 
![alt text][image3]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `def perspect_Transform(img, square = [90,  450, 600, 650])` in `Distort.py`, which appears in the file `Fine_Lanes_main.py`. The `perspect_Transform()` function takes as inputs an image `square` as [width_ylow, low_y, width_ytop, top_y]. 
The result is displayed as a part of the final image. 


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used a function called `slid_window()` in the `Road_lanes.py` file. This calculated the histograms and used the sliding window approach. When the lines were identified I used the function `track_lanes()` to follow the lanes, rather then find them all again every fram. But if the lanes got "wierd" I corrected that by a sanity check and a distance check. All this was put together in the function `get_lane()`. 

See final image to look at the results. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated in the `get_lane()` function in the `Road_lanes.py` file. The following algorithm is used to calculate the curvature: 

`
fit_cr = np.polyfit(yvals*self.ym_per_pix, fitx*self.xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
`

The position of the car was calculated in the `draw_lines()` function in `Road_lanes.py`. The following algorithm was used: 

`
  bottom_leftx = left_fitx[-1]
        bottom_rightx = right_fitx[-1]
        
        lane_center = (bottom_leftx + bottom_rightx) / 2
        
        car_center = 1280 / 2
        
        difference = lane_center - car_center
        
        self.left_lane.line_base_pos  = difference * self.xm_per_pix
        self.right_lane.line_base_pos= self.left_lane.line_base_pos
`
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The finding on the lanes was implemented in the file `Lane_Finding_main.py` with the functions `process_img()` and `get_lane_image()`.  Here is an example of my result on a test image:

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_1.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
