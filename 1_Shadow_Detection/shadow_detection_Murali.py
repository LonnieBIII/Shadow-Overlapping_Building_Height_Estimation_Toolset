# -*- coding: utf-8 -*-

"""
Paper:
S. Murali and V. K. Govindan
Shadow detection and removal from a single image using LAB color space
2013

Code:
Yalım Doğan
https://github.com/YalimD/image_shadow_remover
2018

"""

import cv2 as cv
import os
from osgeo import gdal
import numpy as np
from skimage import measure


#original parameters below

##def detect_shadows(org_image,
##                   lab_adjustment=False,
##                   region_adjustment_kernel_size=10,
##                   shadow_dilation_iteration=5,
##                   shadow_dilation_kernel_size=3,
##                   verbose=False):


def detect_shadows(org_image):

    # If the image is in BGRA color space, convert it to BGR
    if org_image.shape[2] == 4:
        org_image = cv.cvtColor(org_image, cv.COLOR_BGRA2BGR)
    converted_img = cv.cvtColor(org_image, cv.COLOR_BGR2LAB)
    shadow_clear_img = np.copy(org_image)  # Used for constructing corrected image

    # Calculate the mean values of A and B across all pixels
    means = [np.mean(converted_img[:, :, i]) for i in range(3)]
    thresholds = [means[i] - (np.std(converted_img[:, :, i]) / 3) for i in range(3)]

    # If mean is below 256 (which is I think the max value for a channel)
    # Apply threshold using only L
    if sum(means[1:]) <= 256:
        mask = cv.inRange(converted_img, (0, 0, 0), (thresholds[0], 256, 256))
    else:  # Else, also consider B channel
        mask = cv.inRange(converted_img, (0, 0, 0), (thresholds[0], 256, thresholds[2]))

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(mask, kernel, iterations=1)
    #dilation = cv.dilate(erosion, kernel, iterations=1)

    mask = cv.cvtColor(erosion, cv.COLOR_GRAY2RGB)

    
##    cv.imshow("Shadows", mask)
##    
##    cv.imshow("Original Image", org_image)
##
##    cv.waitKey(0)

    return mask
     



def main(input_image, folder_path, base_name):
    org_image = cv.imread(input_image)
    shadow_clear = detect_shadows(org_image)
    cv.imwrite(os.path.join(folder_path, base_name + '_Murali.tif'), shadow_clear)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Murali.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)


