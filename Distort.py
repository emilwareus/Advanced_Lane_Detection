# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 07:29:41 2017

@author: Emil Wåreus
"""


import numpy as np
import cv2


class Distort:
    def corners_unwarp(img, nx, ny, mtx, dist):
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        if ret == True:
            # If we found corners, draw them! (just for fun)
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])
    
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            M_t = cv2.getPerspectiveTransform(dst, src)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)
    
        # Return the resulting image and matrix
        return warped, M, M_t
    
    def unwarp(image, M_t):
        img_size = (image.shape[0], image.shape[1])
        
        unwarped = cv2.warpPerspective(image, M_t, img_size, flags = cv2.INTER_LINEAR)
        
        return unwarped        
        
    def perspect_Transform(img, square = [90,  450, 600, 650]):
        '''
        img is the image you want to transforme
        square contaions 4 variables
        [0] wb = 90
        [1] hb = 450
        [2] wt = 600
        [3] ht = 650
        '''
        H, W = (img.shape[0],img.shape[1])
        wb = square[0]
        hb = square[1]
        wt = square[2]
        ht = square[3]
        
        src = np.float32([[(W/2-wb), hb],[(W/2+wb), hb], [(W/2+wt), ht], [(W/2-wt), ht]])

        img_size = (img.shape[1], img.shape[0])
        offset = 0 #100
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        M_t = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        
        return warped, M, src, M_t
        