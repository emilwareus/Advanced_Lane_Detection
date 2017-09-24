# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:03:23 2017

@author: Emil Wåreus
"""

import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import pickle



image = mpimg.imread('signs_vehicles_xygrad.png')

def dir_threashold(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    
    #1 Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #2 Gradient in X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    #3 Calc dir of gradient
    dir_mask = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    
    
    
    #Create mask
    output = np.zeros_like(dir_mask)
    output[(dir_mask >= thresh[0]) & (dir_mask <= thresh[1])] = 1
    
    return output

def mag_threashold(img, sobel_kernel = 3, thresh = (0, 255)):
    
    #1 Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #2 Gradient in X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    #Magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    #Scale
    scaled_sobel = np.uint8(255*mag/np.max(mag))
    
    
    #Create mask
    output = np.zeros_like(scaled_sobel)
    output[(scaled_sobel>= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return output



def abs_sobel_thresh(img, sobel_kernel=3, orient = 'x', thresh_min = 0, thresh_max = 255):
    #1 Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #2 Gradient in X and Y
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
    #Absolute
    abs_sobel = np.absolute(sobel)
    
    #Scale
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    
    #mask
    
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    

    return mask
    
dir_binary = dir_threashold(image, sobel_kernel=15, thresh=(0.8, 1.2))
gradx = abs_sobel_thresh(image, sobel_kernel=15, orient = 'x', thresh_min = 30, thresh_max = 105)
grady = abs_sobel_thresh(image, sobel_kernel=15, orient = 'y', thresh_min = 30, thresh_max = 105)
mag = mag_threashold(image, sobel_kernel=15, thresh=(80, 155))

combined = np.zeros_like(binary)
combined[((gradx == 1) & (grady == 1)) & ((mag == 1) &(dir_binary == 1))] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize= (24, 9))
f.tight_layout()
ax2.imshow(image)
ax2.set_title('Original Image', fontsize = 50)



ax1.imshow(combined, cmap = 'gray')
ax1.set_title('Thres Image', fontsize = 50)
plt.subplots_adjust(left=0., right = 1, top = 0.9, bottom = 0.)
