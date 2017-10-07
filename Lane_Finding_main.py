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

from Thresh import Thresh
from Distort import Distort
from Sliding_Window import Sliding_Window

def process_img(image):
    combined = Thresh.threshold(image)

    #Perspective change
    per_img, M, sr_c, M_t = Distort.perspect_Transform(image)
    per_img_C, M_C, src, M_t_c = Distort.perspect_Transform(combined)
    
    
    left_fitx, right_fitx, ploty, exist = Sliding_Window.slid_window(per_img_C)
    
    
    img_with_lines = Sliding_Window.draw_lines(image, per_img_C, left_fitx, right_fitx, ploty, M_t_c, exist)
       
    
    
    return img_with_lines,  left_fitx, right_fitx, ploty, exist, combined, per_img_C

def get_lane_image(image):
    '''
    This methode only returns the processed image.
    Note that it wants a BGR and returns a RGB
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_with_lines,  left_fitx, right_fitx, ploty, exist, combined, per_img_C = process_img(image) 
    img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_RGB2BGR)
    return img_with_lines


image = cv2.imread('test_images/test5.jpg') 

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Pipeline
img_with_lines, left_fitx, right_fitx, ploty, exist, combined, per_img_C = process_img(image)
#Thresholds of image


#Plotting 2x2 window grid
f1, (ax1, ax2) = plt.subplots(1, 2, figsize= (24, 9))
f1.tight_layout()
ax1.imshow(combined, cmap = 'gray')
ax1.set_title('Thres Image', fontsize = 50)

ax2.imshow(image)
#ax2.scatter(src[...,0],src[...,1], c='r', s=60)          
ax2.set_title('Original Image', fontsize = 50)

plt.subplots_adjust(left=0., right = 1, top = 0.9, bottom = 0.)

f2, (ax3, ax4) = plt.subplots(1, 2, figsize= (24, 9))

f2.tight_layout()


ax3.imshow(per_img_C, cmap = 'gray')
if exist:
    ax3.plot(left_fitx, ploty, color='yellow')
    ax3.plot(right_fitx, ploty, color='yellow')
ax3.set_title('Thresh image perspective', fontsize = 50)


ax4.imshow(img_with_lines)
ax4.set_title('Image with Lanes', fontsize = 50)

plt.subplots_adjust(left=0., right = 1, top = 0.9, bottom = 0.)



#Lets make some Movies: 

MakeMovie = True

if MakeMovie ==True:
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    
    vid_output = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    vid_clip = clip1.fl_image(get_lane_image)
    vid_clip.write_videofile(vid_output, audio=False)
    
    
