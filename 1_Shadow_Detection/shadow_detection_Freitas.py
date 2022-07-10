#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Paper:
V. L. S. Freitas, B. M. F. Reis, and A. M. G. Tommaselli
Automatic shadow detection in aerial and terrestrial images
2017

Code:
Vander Freitas
https://github.com/vanderfreitas/shadowdetection
2019

"""

import cv2
import os
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.signal import argrelextrema
import scipy.ndimage.filters as ft
import scipy.signal as sg

from matplotlib.widgets import Slider, Button, RadioButtons





def isShadow(K_R, K_G, K_B):

    # Property 1: tal*K_H^80 < K_H < eta*K_H^20
    if K_R < 1.59 or K_R > 48.44:
        return False

    if K_G < 1.43 or K_G > 41.6:
        return False

    if K_B < 1.27 or K_B > 32.4:
        return False

    # Property 2   K_R > K_G > K_B
    if (K_B>K_R) or (K_B>K_G) or (K_G>K_R):
        return False

    # Property 3
    # K_R-K_G>epsilon and K_G-K_B>epsilon  if  K_R>K_R^80
    # K_R-K_G> (epsilon/2) and K_G-K_B>(epsilon/2)  if  K_R<=K_R^80
    # epsilon = 0.33/2 = 0.165

    if K_R>3.18:
        if (K_R-K_G)<0.165 or (K_G-K_B)<0.165:
            return False

    elif (K_R-K_G)<0.0825 or (K_G-K_B)<0.0825:
        return False

    return True


def shadowDetection_Santos_KH(imgIn):

    rows = imgIn.shape[0]
    cols = imgIn.shape[1]

    mask = np.zeros((3,3), dtype=float)

    mask[0,0] = -1.0;
    mask[0,1] = -1.0;
    mask[0,2] = -1.0;
    mask[1,0] = -1.0;
    mask[1,1] = 32.0;
    mask[1,2] = -1.0;
    mask[2,0] = -1.0;
    mask[2,1] = -1.0;
    mask[2,2] = -1.0;


    imgOut = np.zeros((rows, cols), dtype=np.uint8)
    shadowMatrix = np.full((rows, cols), 0, dtype=float)
    I_band = np.zeros((rows, cols), dtype=np.uint8)

    np.zeros((rows, cols), dtype=np.uint8)

    for l in range(rows):
        for c in range(cols):
            R = imgIn.item(l,c,2)
            G = imgIn.item(l,c,1)
            B = imgIn.item(l,c,0)


            # Compute components S and I from HSI system - values between 0 and 255
            I  = (0.33333333) * (R + G + B);

            I_band[l,c] = I


    w = 7
    gray = cv2.bilateralFilter(I_band, w, 700, 25)
    shadowMatrix = sg.convolve(gray,mask)/8.0

    threshold = np.mean(gray)

    R_medio_nonShdw = 0.0
    G_medio_nonShdw = 0.0
    B_medio_nonShdw = 0.0
    count = 0.0
    for l in range(2,rows-2):
        for c in range(2,cols-2):
            # Shadow candidates (value 255) are values below the threshold
            if shadowMatrix[l,c] > threshold*1.3:

                # Compute the average R,G,B values from the original image only on shadow-free areas.
                if gray[l,c] > 255/2:
                    R_medio_nonShdw += imgIn.item(l,c,2)
                    G_medio_nonShdw += imgIn.item(l,c,1)
                    B_medio_nonShdw += imgIn.item(l,c,0)
                    count += 1.0
            else:
                imgOut[l,c] = 255

    # imgOut2 is a copy of imgOut
    imgOut2 = np.copy(imgOut)

    # Just in case there are shadow-free pixels
    if count > 0:
        R_medio_nonShdw = R_medio_nonShdw / count
        G_medio_nonShdw = G_medio_nonShdw / count
        B_medio_nonShdw = B_medio_nonShdw / count

        K_R = 0.0
        K_G = 0.0
        K_B = 0.0

        # Checking on shadow-free areas, verifying the K_H property with windows of size 3
        for l in range(2,rows-2):
            for c in range(2,cols-2):

                R_med = 0.0
                G_med = 0.0
                B_med = 0.0

                # All pixels from the window below must belong to the same target
                # in order to apply the shadow verification process.
                allSameTarget = True
                target = imgOut[l,c]
                for ll in range(-1,2):
                    for cc in range(-1,2):
                        if imgOut[l+ll, c+cc] != target:
                            allSameTarget = False
                        else:
                            B_med += imgIn[l+ll, c+cc, 0]
                            G_med += imgIn[l+ll, c+cc, 1]
                            R_med += imgIn[l+ll, c+cc, 2]


                # Verify the K_H
                # if case they are from the same target
                if allSameTarget == True:
                    # Calcula os componentes K_H
                    R_med = R_med / 9.0
                    G_med = G_med / 9.0
                    B_med = B_med / 9.0

                    K_R=((R_medio_nonShdw+14)**2.4) / ((R_med+14)**2.4)
                    K_G=((G_medio_nonShdw+14)**2.4) / ((G_med+14)**2.4)
                    K_B=((B_medio_nonShdw+14)**2.4) / ((B_med+14)**2.4)

                    # Case the value belong to a shadow, then label it
                    if isShadow(K_R, K_G, K_B) == True:
                        imgOut2[l,c] = 255



    strElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # Structuring element is a circle
    imgOut2 = cv2.morphologyEx(imgOut2, cv2.MORPH_CLOSE, strElem)
    kernel = np.ones((3, 3), np.uint8)
    imgOut2 = cv2.erode(imgOut2, kernel, iterations=1)
    #imgOut2 = cv2.dilate(imgOut2, kernel, iterations=1)

    return imgOut2


def main(input_image, folder_path, base_name):
    imgRef = cv2.imread(input_image)
    imgOut = shadowDetection_Santos_KH(imgRef)
    cv2.imwrite(os.path.join(folder_path, base_name + '_Freitas.tif'), imgOut)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Freitas.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)

