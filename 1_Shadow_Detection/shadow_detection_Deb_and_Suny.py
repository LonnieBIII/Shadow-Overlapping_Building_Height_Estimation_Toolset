"""
Paper:
K. Deb and A. H. Suny
Shadow detection and removal based on YCbCr color space
2014

Code:
Mykhailo Mostipan
https://github.com/mykhailo-mostipan/shadow-removal
2018

"""

import numpy as np
import cv2
import os
from osgeo import gdal


def main(input_image, folder_path, base_name):
    # read an image with shadow...
    # and it converts to BGR color space automatically
    or_img = cv2.imread(input_image)

    # covert the BGR image to an YCbCr image
    y_cb_cr_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2YCrCb)

    # copy the image to create a binary mask later
    binary_mask = np.copy(y_cb_cr_img)

    # get mean value of the pixels in Y plane
    y_mean = np.mean(cv2.split(y_cb_cr_img)[0])

    # get standard deviation of channel in Y plane
    y_std = np.std(cv2.split(y_cb_cr_img)[0])

    # classify pixels as shadow and non-shadow pixels
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):

            if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
                # paint it white (shadow)
                binary_mask[i, j] = [255, 255, 255]
            else:
                # paint it black (non-shadow)
                binary_mask[i, j] = [0, 0, 0]

    # Using morphological operation
    # The misclassified pixels are
    # removed using erosion followed by dilation.
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_mask, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)



    ##cv2.imshow("im1", or_img)
    ##cv2.imshow("im2", erosion)
    ##cv2.waitKey(0)
    ##cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(folder_path, base_name + '_Deb_and_Suny.tif'), erosion)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Deb_and_Suny.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)
